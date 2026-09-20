"""Expanded Phase 2 collection protocol and dependency-light validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

import numpy as np


COLLECTION_SCHEMA_VERSION = "pilot_v0_3_expanded_collection_v1"
PILOT_VERSION = "0.3"
PROTOCOL_ID = "pilot-v0.3-expanded-v1"
SPLIT_RULE_ID = "pilot-v0.3-expanded-split-v1"
TASK_IDS = tuple(range(10))
TARGET_PROGRESS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.90)
HELDOUT_FRACTION = 0.20
CHECKPOINT_IDENTITY = "openvla/openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial_no_noops"
GLOBAL_SEED = 7
PROGRESS_SCHEMA_VERSION = "pilot_v0_3_collection_progress_v1"

_SAMPLE_ID_PATTERN = re.compile(
    r"libero_spatial__task(?P<task>\d{2})__state(?P<state>\d{2})"
    r"__step(?P<step>\d{4})"
)


class ExpandedDatasetError(RuntimeError):
    """Raised when the expanded Phase 2 dataset contract is violated."""


@dataclass(frozen=True)
class SourceRecord:
    sample_id: str
    source_observation_path: str
    resolved_observation_path: Path
    source_image_hash: str


@dataclass(frozen=True)
class SourceCollection:
    manifest_path: Path
    checkpoint_identity: str
    libero_revision: str
    records: tuple[SourceRecord, ...]
    dataset_protocol: dict[str, Any]


@dataclass(frozen=True)
class ExpandedCollectionResult:
    manifest_path: Path
    sample_paths: tuple[Path, ...]
    status: str
    split_readiness: str


class _FallbackProgress:
    """Dependency-free state progress used when tqdm is unavailable."""

    def __init__(self, *, total: int, initial: int = 0, desc: str) -> None:
        self.total = total
        self.n = initial
        self.initial = initial
        self.desc = desc
        self.started = time.monotonic()
        self.postfix: dict[str, Any] = {}

    def set_postfix(self, values: dict[str, Any], *, refresh: bool = False) -> None:
        self.postfix = dict(values)
        if refresh:
            self.refresh()

    def update(self, count: int = 1) -> None:
        self.n += count
        self.refresh()

    def refresh(self) -> None:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        session_completed = max(self.n - self.initial, 0)
        rate = session_completed / elapsed
        remaining = max(self.total - self.n, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        fields = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "unknown"
        print(
            f"{self.desc}: {self.n}/{self.total} {fields} "
            f"elapsed={elapsed:.1f}s rate={rate:.3f}state/s ETA={eta_text}",
            flush=True,
        )

    def write(self, message: str) -> None:
        print(message, flush=True)

    def close(self) -> None:
        sys.stdout.flush()


def _default_progress_factory(*, total: int, initial: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(
            "tqdm is unavailable; using the dependency-free flushed ETA reporter",
            flush=True,
        )
        return _FallbackProgress(
            total=total,
            initial=initial,
            desc="Expanded Phase2 collection",
        )
    return tqdm(
        total=total,
        initial=initial,
        desc="Expanded Phase2 collection",
        unit="state",
        dynamic_ncols=True,
        mininterval=0.5,
    )


def _progress_log(progress: Any | None, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
        sys.stdout.flush()
    else:
        print(message, flush=True)


def sample_indices(
    trajectory_length: int,
    target_progress: Sequence[float] = TARGET_PROGRESS,
) -> tuple[int, ...]:
    if type(trajectory_length) is not int or trajectory_length < 2:
        raise ExpandedDatasetError("trajectory_length must be an integer >= 2")
    progress = tuple(float(value) for value in target_progress)
    if (
        not progress
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in progress)
        or tuple(sorted(progress)) != progress
        or len(set(progress)) != len(progress)
    ):
        raise ExpandedDatasetError(
            "target progress must be finite, unique, and sorted within [0,1]"
        )
    return tuple(
        math.floor(value * (trajectory_length - 1) + 0.5) for value in progress
    )


def capacity_report(task_suite: Any, *, libero_revision: str) -> dict[str, Any]:
    if not isinstance(libero_revision, str) or not libero_revision.strip():
        raise ExpandedDatasetError("LIBERO revision must be a non-empty string")
    task_count = task_suite.get_num_tasks()
    if task_count != len(TASK_IDS):
        raise ExpandedDatasetError(
            f"expected 10 LIBERO-Spatial tasks, got {task_count!r}"
        )
    per_task = {
        str(task_id): len(task_suite.get_task_init_states(task_id))
        for task_id in TASK_IDS
    }
    if any(type(count) is not int or count <= 0 for count in per_task.values()):
        raise ExpandedDatasetError("every task must expose at least one initial state")
    return {
        "schema_version": "pilot_v0_3_initial_state_capacity_audit_v1",
        "status": "CAPACITY_AUDIT_COMPLETE",
        "suite": "libero_spatial",
        "libero_revision": libero_revision,
        "task_ids": list(TASK_IDS),
        "available_initial_states_per_task": per_task,
        "maximum_unique_groups": sum(per_task.values()),
        "group_identity_fields": ["task_id", "initial_state_id"],
    }


def _protocol_metadata() -> dict[str, Any]:
    return {
        "task_ids": list(TASK_IDS),
        "state_selection": "all_official_initial_states_ascending",
        "observations_per_group": len(TARGET_PROGRESS),
        "target_relative_progress": list(TARGET_PROGRESS),
        "sampling_rounding": "floor(q*(T-1)+0.5)",
        "reject_duplicate_timesteps": True,
        "group_identity_fields": ["task_id", "initial_state_id"],
        "sample_identity_fields": [
            "task_id",
            "initial_state_id",
            "target_relative_progress",
        ],
        "split_rule_id": SPLIT_RULE_ID,
        "heldout_fraction_per_task": HELDOUT_FRACTION,
        "resume_enabled": False,
    }


def _new_task_results(suite: Any, capacity: dict[str, int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        task = suite.get_task(task_id)
        if not isinstance(task.language, str) or not task.language:
            raise ExpandedDatasetError(f"task {task_id} language is empty")
        results.append(
            {
                "task_id": task_id,
                "task_language": task.language,
                "available_initial_states": capacity[str(task_id)],
                "attempted_states": 0,
                "successful_clean_openvla_trajectories": 0,
                "failed_trajectories": 0,
                "accepted_unique_groups": 0,
                "attempted_state_ids": [],
                "accepted_state_ids": [],
                "failed_states": [],
                "accepted_groups": [],
            }
        )
    return results


def _apply_completed_state(
    task_results: list[dict[str, Any]], state: dict[str, Any]
) -> None:
    task_id = state["task_id"]
    state_id = state["initial_state_id"]
    result = task_results[task_id]
    if state_id in result["attempted_state_ids"]:
        raise ExpandedDatasetError(f"duplicate completed state: {(task_id, state_id)}")
    result["attempted_state_ids"].append(state_id)
    result["attempted_states"] += 1
    outcome = state["outcome"]
    if outcome == "accepted":
        result["successful_clean_openvla_trajectories"] += 1
        result["accepted_unique_groups"] += 1
        result["accepted_state_ids"].append(state_id)
        result["accepted_groups"].append(state["group"])
    elif outcome == "policy_failure":
        result["failed_trajectories"] += 1
        result["failed_states"].append(
            {
                "initial_state_id": state_id,
                "reason": "policy_failure",
                "trajectory_length": state["trajectory_length"],
            }
        )
    elif outcome == "sampling_rejected":
        result["successful_clean_openvla_trajectories"] += 1
        result["failed_states"].append(
            {
                "initial_state_id": state_id,
                "reason": "sampling_index_collision",
                "trajectory_length": state["trajectory_length"],
            }
        )
    else:
        raise ExpandedDatasetError(f"invalid completed-state outcome: {outcome!r}")


def _progress_counts(states: Sequence[dict[str, Any]]) -> dict[str, int]:
    accepted = sum(state["outcome"] == "accepted" for state in states)
    policy_failed = sum(state["outcome"] == "policy_failure" for state in states)
    sampling_rejected = sum(state["outcome"] == "sampling_rejected" for state in states)
    observations = sum(
        len(state["group"]["samples"])
        for state in states
        if state["outcome"] == "accepted"
    )
    return {
        "completed": len(states),
        "accepted": accepted,
        "policy_failed": policy_failed,
        "sampling_rejected": sampling_rejected,
        "observation_files": observations,
    }


def _progress_summary(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accepted_states = [
        {"task_id": state["task_id"], "initial_state_id": state["initial_state_id"]}
        for state in states
        if state["outcome"] == "accepted"
    ]
    policy_failed_states = [
        {"task_id": state["task_id"], "initial_state_id": state["initial_state_id"]}
        for state in states
        if state["outcome"] == "policy_failure"
    ]
    sampling_rejected_states = [
        {"task_id": state["task_id"], "initial_state_id": state["initial_state_id"]}
        for state in states
        if state["outcome"] == "sampling_rejected"
    ]
    observation_files = [
        {
            "task_id": state["task_id"],
            "initial_state_id": state["initial_state_id"],
            "sample_id": sample["sample_id"],
            "observation_path": sample["observation_path"],
            "sha256": sample["sha256"],
        }
        for state in states
        if state["outcome"] == "accepted"
        for sample in state["group"]["samples"]
    ]
    return {
        "completed_states": [
            {
                "task_id": state["task_id"],
                "initial_state_id": state["initial_state_id"],
                "outcome": state["outcome"],
            }
            for state in states
        ],
        "accepted_states": accepted_states,
        "policy_failed_states": policy_failed_states,
        "sampling_rejected_states": sampling_rejected_states,
        "observation_files": observation_files,
        "counts": _progress_counts(states),
    }


def _save_progress(
    path: Path,
    progress: dict[str, Any],
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    progress["status"] = status
    progress.update(_progress_summary(progress["states"]))
    progress["last_error"] = (
        None
        if error is None
        else {"error_type": type(error).__name__, "message": str(error)}
    )
    _atomic_json(path, progress)


def _load_resume_progress(
    *,
    progress_path: Path,
    expected_identity: dict[str, Any],
    state_order: Sequence[tuple[int, int]],
    observations_dir: Path,
    observation_type: Any,
) -> dict[str, Any]:
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ExpandedDatasetError("failed to load collection progress") from error
    for key, expected in expected_identity.items():
        if progress.get(key) != expected:
            raise ExpandedDatasetError(f"resume identity mismatch: {key}")
    if progress.get("status") not in {
        "IN_PROGRESS",
        "INTERRUPTED_RESUMABLE",
        "FINALIZING",
    }:
        raise ExpandedDatasetError(
            f"collection progress is not resumable: {progress.get('status')!r}"
        )
    execution = progress.get("execution")
    if (
        not isinstance(execution, dict)
        or type(execution.get("resume_count")) is not int
        or execution["resume_count"] < 0
    ):
        raise ExpandedDatasetError("invalid resume execution metadata")
    states = progress.get("states")
    if not isinstance(states, list) or len(states) > len(state_order):
        raise ExpandedDatasetError("invalid completed-state records")

    sample_ids: set[str] = set()
    identities: set[tuple[int, int, float]] = set()
    for index, state in enumerate(states):
        if not isinstance(state, dict):
            raise ExpandedDatasetError("completed state must be an object")
        pair = (state.get("task_id"), state.get("initial_state_id"))
        if pair != state_order[index]:
            raise ExpandedDatasetError(
                "completed states must be the canonical task/state prefix"
            )
        trajectory_length = state.get("trajectory_length")
        if type(trajectory_length) is not int or trajectory_length < 0:
            raise ExpandedDatasetError(f"invalid trajectory length: {pair}")
        outcome = state.get("outcome")
        if outcome not in {"accepted", "policy_failure", "sampling_rejected"}:
            raise ExpandedDatasetError(f"invalid completed-state outcome: {outcome}")
        group = state.get("group")
        if outcome != "accepted":
            if group is not None:
                raise ExpandedDatasetError("rejected completed state contains a group")
            if outcome == "sampling_rejected":
                if trajectory_length <= 1 or len(
                    set(sample_indices(trajectory_length))
                ) == len(TARGET_PROGRESS):
                    raise ExpandedDatasetError(
                        f"invalid sampling-rejected state: {pair}"
                    )
            continue
        if (
            not isinstance(group, dict)
            or group.get("task_id") != pair[0]
            or group.get("initial_state_id") != pair[1]
            or group.get("trajectory_length") != trajectory_length
            or group.get("episode_success") is not True
            or not isinstance(group.get("samples"), list)
            or len(group["samples"]) != len(TARGET_PROGRESS)
        ):
            raise ExpandedDatasetError(f"invalid completed group: {pair}")
        if any(not isinstance(sample, dict) for sample in group["samples"]):
            raise ExpandedDatasetError(f"invalid completed sample: {pair}")
        if [
            sample.get("target_relative_progress") for sample in group["samples"]
        ] != list(TARGET_PROGRESS):
            raise ExpandedDatasetError(f"completed group progress differs: {pair}")
        step_ids = [sample.get("step_id") for sample in group["samples"]]
        if (
            trajectory_length <= 1
            or any(type(step_id) is not int for step_id in step_ids)
            or any(not 0 <= step_id < trajectory_length for step_id in step_ids)
            or len(set(step_ids)) != len(step_ids)
            or tuple(step_ids) != sample_indices(trajectory_length)
        ):
            raise ExpandedDatasetError(f"completed group timesteps differ: {pair}")
        for sample, step_id in zip(group["samples"], step_ids, strict=True):
            actual_progress = sample.get("actual_normalized_episode_progress")
            if not isinstance(actual_progress, (int, float)) or not math.isclose(
                actual_progress,
                step_id / (trajectory_length - 1),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ExpandedDatasetError(f"completed sample progress differs: {pair}")
            record = _validate_sample(
                sample,
                task_id=pair[0],
                state_id=pair[1],
                manifest_parent=progress_path.parent,
                observations_dir=observations_dir,
                observation_type=observation_type,
            )
            identity = (
                pair[0],
                pair[1],
                float(sample["target_relative_progress"]),
            )
            if record.sample_id in sample_ids or identity in identities:
                raise ExpandedDatasetError("duplicate sample identity in resume state")
            sample_ids.add(record.sample_id)
            identities.add(identity)

    expected_summary = _progress_summary(states)
    for key, expected in expected_summary.items():
        if progress.get(key) != expected:
            raise ExpandedDatasetError(f"resume checkpoint summary mismatch: {key}")
    return progress


def _remove_uncheckpointed_observations(
    observations_dir: Path, states: Sequence[dict[str, Any]]
) -> None:
    referenced = {
        (observations_dir.parent / sample["observation_path"]).resolve()
        for state in states
        if state["outcome"] == "accepted"
        for sample in state["group"]["samples"]
    }
    for path in observations_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ExpandedDatasetError(
                f"unexpected entry in observations directory: {path}"
            )
        if path.resolve() not in referenced:
            path.unlink()


def _prepare_resume_final_outputs(
    destination: Path, *, completed_states: int, total_states: int
) -> None:
    """Discard only final outputs left by interrupted finalization."""

    final_paths = (
        destination / "collection_manifest.json",
        destination / "feasibility_report.json",
    )
    existing = [path for path in final_paths if path.exists()]
    if existing and completed_states != total_states:
        raise ExpandedDatasetError(
            "incomplete checkpoint unexpectedly contains formal final outputs"
        )
    _remove_formal_outputs(destination)
    temporary = destination / "collection_progress.json.tmp"
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise ExpandedDatasetError(f"invalid progress temporary path: {temporary}")
        temporary.unlink()


def _remove_formal_outputs(destination: Path) -> None:
    for path in (
        destination / "collection_manifest.json",
        destination / "feasibility_report.json",
    ):
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ExpandedDatasetError(f"invalid formal output path: {path}")
        path.unlink()


def collect_expanded_observations(
    *,
    model: Any,
    processor: Any,
    pretrained_checkpoint: str | Path,
    libero_revision: str,
    code_commit: str,
    shared_feature_commit: str,
    output_dir: str | Path,
    resume: bool = False,
    progress_factory: Callable[..., Any] | None = None,
    _collector: Any | None = None,
    _runtime: Any | None = None,
    _suite: Any | None = None,
    _observation_type: Any | None = None,
) -> ExpandedCollectionResult:
    """Attempt every official state and persist every valid successful trajectory."""

    if _collector is None or _observation_type is None:
        from shared_feature import PilotObservation
        from shared_feature import libero_collector as collector

        _collector = collector if _collector is None else _collector
        _observation_type = (
            PilotObservation if _observation_type is None else _observation_type
        )
    collector = _collector
    observation_type = _observation_type

    checkpoint = Path(pretrained_checkpoint).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ExpandedDatasetError(f"checkpoint is not a directory: {checkpoint}")
    if resume:
        if not destination.is_dir():
            raise ExpandedDatasetError(
                f"resume output directory does not exist: {destination}"
            )
    elif destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ExpandedDatasetError(f"output directory must be fresh: {destination}")
    runtime = _runtime or collector._load_official_runtime()
    suite = _suite or runtime.benchmark.get_benchmark_dict()[collector.PILOT_SUITE]()
    audit = capacity_report(suite, libero_revision=libero_revision)
    for name, commit in {
        "code_commit": code_commit,
        "shared_feature_commit": shared_feature_commit,
    }.items():
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ExpandedDatasetError(f"{name} must be a full lowercase Git SHA")

    capacity = audit["available_initial_states_per_task"]
    state_order = tuple(
        (task_id, state_id)
        for task_id in TASK_IDS
        for state_id in range(capacity[str(task_id)])
    )
    total_states = len(state_order)
    protocol = _protocol_metadata()
    rollout_identity = {
        "checkpoint_identity": CHECKPOINT_IDENTITY,
        "resolved_checkpoint_path": str(checkpoint),
        "unnorm_key": UNNORM_KEY,
        "center_crop": True,
        "camera_resolution": collector._CAMERA_RESOLUTION,
        "dummy_steps": collector._NUM_DUMMY_STEPS,
        "max_valid_policy_actions": collector._MAX_POLICY_ACTIONS,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "global_seed": GLOBAL_SEED,
        "environment_seed": 0,
        "do_sample": False,
    }
    expected_identity = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol,
        "provenance": {
            "code_commit": code_commit,
            "shared_feature_commit": shared_feature_commit,
            "libero_revision": libero_revision,
        },
        "rollout": rollout_identity,
        "capacity": {
            "task_ids": list(TASK_IDS),
            "available_initial_states_per_task": capacity,
            "total_state_attempts": total_states,
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    observations_dir = destination / "observations"
    progress_path = destination / "collection_progress.json"
    if resume:
        if not observations_dir.is_dir() or not progress_path.is_file():
            raise ExpandedDatasetError(
                "resume requires observations/ and collection_progress.json"
            )
        progress_state = _load_resume_progress(
            progress_path=progress_path,
            expected_identity=expected_identity,
            state_order=state_order,
            observations_dir=observations_dir,
            observation_type=observation_type,
        )
        progress_state["execution"]["resume_count"] += 1
        progress_state["execution"]["resumed"] = True
        _remove_uncheckpointed_observations(
            observations_dir,
            progress_state["states"],
        )
        _prepare_resume_final_outputs(
            destination,
            completed_states=len(progress_state["states"]),
            total_states=total_states,
        )
    else:
        observations_dir.mkdir()
        progress_state = {
            **expected_identity,
            "status": "IN_PROGRESS",
            "execution": {"resumed": False, "resume_count": 0},
            "completed_states": [],
            "accepted_states": [],
            "policy_failed_states": [],
            "sampling_rejected_states": [],
            "observation_files": [],
            "counts": {},
            "states": [],
            "last_error": None,
        }
        _save_progress(progress_path, progress_state, status="IN_PROGRESS")

    task_results = _new_task_results(suite, capacity)
    sample_paths: list[Path] = []
    sample_ids: set[str] = set()
    sample_identities: set[tuple[int, int, float]] = set()
    group_ids: set[tuple[int, int]] = set()
    completed_pairs: set[tuple[int, int]] = set()
    for state in progress_state["states"]:
        _apply_completed_state(task_results, state)
        completed_pairs.add((state["task_id"], state["initial_state_id"]))
        if state["outcome"] != "accepted":
            continue
        group = state["group"]
        group_id = (group["task_id"], group["initial_state_id"])
        group_ids.add(group_id)
        for sample in group["samples"]:
            sample_ids.add(sample["sample_id"])
            sample_identities.add(
                (
                    group["task_id"],
                    group["initial_state_id"],
                    float(sample["target_relative_progress"]),
                )
            )
            sample_paths.append(
                (destination / sample["observation_path"]).resolve(strict=True)
            )

    action_config = collector._OpenVLAActionConfig(
        pretrained_checkpoint=str(checkpoint),
        unnorm_key=UNNORM_KEY,
        center_crop=True,
    )
    progress = None

    try:
        initial = len(progress_state["states"])
        progress = (progress_factory or _default_progress_factory)(
            total=total_states,
            initial=initial,
        )
        _progress_log(
            progress,
            f"collection start: completed={initial}/{total_states} resume={resume}",
        )
        for task_id in TASK_IDS:
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            if not isinstance(task.language, str) or not task.language:
                raise ExpandedDatasetError(f"task {task_id} language is empty")
            _progress_log(
                progress,
                f"task transition: task={task_id} states={len(states)}",
            )
            for state_id, initial_state in enumerate(states):
                if (task_id, state_id) in completed_pairs:
                    continue
                counts = _progress_counts(progress_state["states"])
                progress.set_postfix(
                    {
                        "task": task_id,
                        "state": state_id,
                        "accepted": counts["accepted"],
                        "policy_failed": counts["policy_failed"],
                        "sampling_rejected": counts["sampling_rejected"],
                    },
                    refresh=True,
                )
                try:
                    trajectory, success = collector._collect_episode(
                        runtime=runtime,
                        action_config=action_config,
                        model=model,
                        processor=processor,
                        task=task,
                        initial_state=initial_state,
                        initial_state_id=state_id,
                        task_id=task_id,
                    )
                except collector._EpisodeCollectionError as error:
                    raise ExpandedDatasetError(
                        f"rollout infrastructure failure for task={task_id}, "
                        f"state={state_id}, category={error.category}"
                    ) from error
                if not success:
                    state_record = {
                        "task_id": task_id,
                        "initial_state_id": state_id,
                        "trajectory_length": len(trajectory),
                        "outcome": "policy_failure",
                        "group": None,
                    }
                else:
                    indices = sample_indices(len(trajectory))
                    if len(set(indices)) != len(TARGET_PROGRESS):
                        state_record = {
                            "task_id": task_id,
                            "initial_state_id": state_id,
                            "trajectory_length": len(trajectory),
                            "outcome": "sampling_rejected",
                            "group": None,
                        }
                    else:
                        group_id = (task_id, state_id)
                        if group_id in group_ids:
                            raise ExpandedDatasetError(
                                f"duplicate trajectory group: {group_id}"
                            )
                        records: list[tuple[Any, Path]] = []
                        samples: list[dict[str, Any]] = []
                        for target_progress, index in zip(
                            TARGET_PROGRESS, indices, strict=True
                        ):
                            buffered = trajectory[index]
                            sample_id = (
                                f"libero_spatial__task{task_id:02d}"
                                f"__state{state_id:02d}"
                                f"__step{buffered.step_id:04d}"
                            )
                            identity = (task_id, state_id, target_progress)
                            if sample_id in sample_ids or identity in sample_identities:
                                raise ExpandedDatasetError(
                                    "duplicate sample identity: "
                                    f"{sample_id}, {identity}"
                                )
                            path = observations_dir / f"{sample_id}.npz"
                            observation = observation_type(
                                sample_id=sample_id,
                                task_id=str(task_id),
                                initial_state_id=state_id,
                                episode_id=state_id,
                                step_id=buffered.step_id,
                                normalized_episode_progress=(
                                    buffered.step_id / (len(trajectory) - 1)
                                ),
                                base_rgb_raw=buffered.base_rgb_raw,
                                wrist_rgb_raw=buffered.wrist_rgb_raw,
                                state=buffered.state,
                                prompt=task.language,
                                episode_success=True,
                            )
                            records.append((observation, path))
                            samples.append(
                                {
                                    "sample_id": sample_id,
                                    "step_id": buffered.step_id,
                                    "target_relative_progress": target_progress,
                                    "actual_normalized_episode_progress": (
                                        buffered.step_id / (len(trajectory) - 1)
                                    ),
                                    "observation_path": (
                                        f"observations/{sample_id}.npz"
                                    ),
                                }
                            )
                        committed = _commit_group(records)
                        for sample, path in zip(samples, committed, strict=True):
                            sample["sha256"] = _sha256_file(path)
                        group = {
                            "task_id": task_id,
                            "initial_state_id": state_id,
                            "trajectory_length": len(trajectory),
                            "episode_success": True,
                            "samples": samples,
                        }
                        state_record = {
                            "task_id": task_id,
                            "initial_state_id": state_id,
                            "trajectory_length": len(trajectory),
                            "outcome": "accepted",
                            "group": group,
                        }
                        sample_paths.extend(committed)
                        sample_ids.update(record.sample_id for record, _ in records)
                        sample_identities.update(
                            (task_id, state_id, value) for value in TARGET_PROGRESS
                        )
                        group_ids.add(group_id)

                progress_state["states"].append(state_record)
                completed_pairs.add((task_id, state_id))
                _apply_completed_state(task_results, state_record)
                _save_progress(progress_path, progress_state, status="IN_PROGRESS")
                counts = _progress_counts(progress_state["states"])
                progress.set_postfix(
                    {
                        "task": task_id,
                        "state": state_id,
                        "accepted": counts["accepted"],
                        "policy_failed": counts["policy_failed"],
                        "sampling_rejected": counts["sampling_rejected"],
                    },
                    refresh=False,
                )
                progress.update(1)
                _progress_log(
                    progress,
                    "checkpoint saved: "
                    f"completed={counts['completed']}/{total_states} "
                    f"task={task_id} state={state_id}",
                )

        _save_progress(progress_path, progress_state, status="FINALIZING")
        accepted = {
            str(row["task_id"]): row["accepted_unique_groups"] for row in task_results
        }
        readiness = (
            "READY_FOR_SPLIT"
            if all(count >= 2 for count in accepted.values())
            else "INSUFFICIENT_FOR_SPLIT"
        )
        total_groups = sum(accepted.values())
        manifest = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "pilot_version": PILOT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "suite": "libero_spatial",
            "run_status": "COMPLETED",
            "split_readiness": readiness,
            "execution": {
                "resumed": progress_state["execution"]["resumed"],
                "resume_count": progress_state["execution"]["resume_count"],
                "completed_state_count": len(progress_state["states"]),
                "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            },
            "provenance": {
                "code_commit": code_commit,
                "shared_feature_commit": shared_feature_commit,
                "libero_revision": libero_revision,
            },
            "rollout": {
                "policy_family": "openvla",
                **rollout_identity,
            },
            "protocol": protocol,
            "runtime": {
                "libero_revision": libero_revision,
                "discovered_task_count": len(TASK_IDS),
                "task_ids": list(TASK_IDS),
                "official_initial_state_counts": audit[
                    "available_initial_states_per_task"
                ],
            },
            "feasibility": {
                "available_initial_states_per_task": audit[
                    "available_initial_states_per_task"
                ],
                "attempted_states_per_task": {
                    str(row["task_id"]): row["attempted_states"] for row in task_results
                },
                "successful_clean_trajectories_per_task": {
                    str(row["task_id"]): row["successful_clean_openvla_trajectories"]
                    for row in task_results
                },
                "failed_trajectories_per_task": {
                    str(row["task_id"]): row["failed_trajectories"]
                    for row in task_results
                },
                "accepted_unique_groups_per_task": accepted,
                "maximum_available_unique_groups": audit["maximum_unique_groups"],
                "actual_accepted_unique_groups": total_groups,
            },
            "coverage": {
                "actual_total_groups": total_groups,
                "actual_total_observations": total_groups * len(TARGET_PROGRESS),
                "accepted_groups_per_task": accepted,
            },
            "task_results": task_results,
        }
        manifest_path = destination / "collection_manifest.json"
        _atomic_json(manifest_path, manifest)
        _atomic_json(
            destination / "feasibility_report.json",
            _feasibility_report(manifest),
        )
        validate_expanded_collection(
            manifest_path,
            observation_type=observation_type,
        )
        _save_progress(progress_path, progress_state, status="COMPLETE")
        _progress_log(
            progress,
            "collection completed: "
            f"states={total_states} accepted={total_groups} observations={len(sample_paths)}",
        )
    except KeyboardInterrupt as error:
        _remove_formal_outputs(destination)
        _save_progress(
            progress_path,
            progress_state,
            status="INTERRUPTED_RESUMABLE",
            error=error,
        )
        _progress_log(
            progress,
            "failure summary: status=INTERRUPTED_RESUMABLE "
            f"completed={len(progress_state['states'])}/{total_states}",
        )
        raise
    except BaseException as error:
        _remove_formal_outputs(destination)
        _save_progress(
            progress_path,
            progress_state,
            status="INTERRUPTED_RESUMABLE",
            error=error,
        )
        _progress_log(
            progress,
            "failure summary: status=INTERRUPTED_RESUMABLE "
            f"error={type(error).__name__}: {error}",
        )
        raise
    finally:
        if progress is not None:
            progress.close()

    return ExpandedCollectionResult(
        manifest_path=manifest_path.resolve(),
        sample_paths=tuple(path.resolve() for path in sample_paths),
        status="EXPANDED_COLLECTION_COMPLETE",
        split_readiness=readiness,
    )


def validate_expanded_collection(
    path: str | Path, *, observation_type: Any | None = None
) -> SourceCollection:
    """Load a completed expanded collection with identity and archive checks."""

    if observation_type is None:
        from shared_feature import PilotObservation

        observation_type = PilotObservation

    manifest_path = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != COLLECTION_SCHEMA_VERSION
        or manifest.get("pilot_version") != PILOT_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("suite") != "libero_spatial"
        or manifest.get("run_status") != "COMPLETED"
    ):
        raise ExpandedDatasetError("invalid expanded collection identity/status")
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("task_ids") != list(TASK_IDS)
        or protocol.get("observations_per_group") != len(TARGET_PROGRESS)
        or protocol.get("target_relative_progress") != list(TARGET_PROGRESS)
        or protocol.get("group_identity_fields") != ["task_id", "initial_state_id"]
        or protocol.get("split_rule_id") != SPLIT_RULE_ID
        or protocol.get("heldout_fraction_per_task") != HELDOUT_FRACTION
    ):
        raise ExpandedDatasetError("expanded collection protocol differs")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "libero_revision"
    ) != manifest.get("runtime", {}).get("libero_revision"):
        raise ExpandedDatasetError("expanded collection provenance differs")
    for key in ("code_commit", "shared_feature_commit"):
        commit = provenance.get(key)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ExpandedDatasetError(f"invalid expanded provenance {key}")
    rollout = manifest.get("rollout")
    if (
        not isinstance(rollout, dict)
        or rollout.get("checkpoint_identity") != CHECKPOINT_IDENTITY
    ):
        raise ExpandedDatasetError("expanded collection checkpoint differs")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("discovered_task_count") != len(TASK_IDS)
        or runtime.get("task_ids") != list(TASK_IDS)
        or not isinstance(runtime.get("libero_revision"), str)
    ):
        raise ExpandedDatasetError("expanded collection runtime differs")
    task_results = manifest.get("task_results")
    if not isinstance(task_results, list) or [
        row.get("task_id") for row in task_results if isinstance(row, dict)
    ] != list(TASK_IDS):
        raise ExpandedDatasetError("expanded task results must be ordered 0..9")

    observations_dir = (manifest_path.parent / "observations").resolve(strict=True)
    records: list[SourceRecord] = []
    sample_ids: set[str] = set()
    sample_identities: set[tuple[int, int, float]] = set()
    group_ids: set[tuple[int, int]] = set()
    resolved_paths: set[Path] = set()
    accepted_per_task: dict[str, int] = {}
    for task_id, row in zip(TASK_IDS, task_results, strict=True):
        available = row.get("available_initial_states")
        attempted = row.get("attempted_state_ids")
        groups = row.get("accepted_groups")
        accepted_state_ids = row.get("accepted_state_ids")
        successful = row.get("successful_clean_openvla_trajectories")
        failed = row.get("failed_trajectories")
        if (
            type(available) is not int
            or available <= 0
            or attempted != list(range(available))
            or row.get("attempted_states") != available
            or type(successful) is not int
            or type(failed) is not int
            or successful + failed != available
            or not isinstance(groups, list)
            or row.get("accepted_unique_groups") != len(groups)
            or accepted_state_ids != [group.get("initial_state_id") for group in groups]
            or len(groups) > successful
        ):
            raise ExpandedDatasetError(f"invalid feasibility counts for task {task_id}")
        accepted_per_task[str(task_id)] = len(groups)
        for group in groups:
            state_id = group.get("initial_state_id")
            group_id = (task_id, state_id)
            samples = group.get("samples")
            if (
                group.get("task_id") != task_id
                or type(state_id) is not int
                or not 0 <= state_id < available
                or group.get("episode_success") is not True
                or group_id in group_ids
                or not isinstance(samples, list)
                or len(samples) != len(TARGET_PROGRESS)
            ):
                raise ExpandedDatasetError(f"malformed expanded group: {group_id}")
            group_ids.add(group_id)
            if [sample.get("target_relative_progress") for sample in samples] != list(
                TARGET_PROGRESS
            ):
                raise ExpandedDatasetError(f"progress set differs for group {group_id}")
            step_ids = [sample.get("step_id") for sample in samples]
            if len(set(step_ids)) != len(TARGET_PROGRESS):
                raise ExpandedDatasetError(f"duplicate sampled timestep: {group_id}")
            for sample in samples:
                record = _validate_sample(
                    sample,
                    task_id=task_id,
                    state_id=state_id,
                    manifest_parent=manifest_path.parent,
                    observations_dir=observations_dir,
                    observation_type=observation_type,
                )
                identity = (
                    task_id,
                    state_id,
                    float(sample["target_relative_progress"]),
                )
                if (
                    record.sample_id in sample_ids
                    or identity in sample_identities
                    or record.resolved_observation_path in resolved_paths
                ):
                    raise ExpandedDatasetError(
                        "duplicate expanded sample identity/path"
                    )
                sample_ids.add(record.sample_id)
                sample_identities.add(identity)
                resolved_paths.add(record.resolved_observation_path)
                records.append(record)

    coverage = manifest.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("accepted_groups_per_task") != accepted_per_task
        or coverage.get("actual_total_groups") != len(group_ids)
        or coverage.get("actual_total_observations") != len(records)
        or len(records) != len(group_ids) * len(TARGET_PROGRESS)
    ):
        raise ExpandedDatasetError("expanded collection coverage differs")
    expected_readiness = (
        "READY_FOR_SPLIT"
        if all(count >= 2 for count in accepted_per_task.values())
        else "INSUFFICIENT_FOR_SPLIT"
    )
    if manifest.get("split_readiness") != expected_readiness:
        raise ExpandedDatasetError("expanded split readiness differs")
    direct_paths = {candidate.resolve() for candidate in observations_dir.glob("*.npz")}
    if direct_paths != resolved_paths:
        raise ExpandedDatasetError("observation archives differ from expanded manifest")
    feasibility_path = manifest_path.parent / "feasibility_report.json"
    if not feasibility_path.is_file() or json.loads(
        feasibility_path.read_text(encoding="utf-8")
    ) != _feasibility_report(manifest):
        raise ExpandedDatasetError(
            "feasibility report differs from collection manifest"
        )

    return SourceCollection(
        manifest_path=manifest_path,
        checkpoint_identity=CHECKPOINT_IDENTITY,
        libero_revision=runtime["libero_revision"],
        records=tuple(records),
        dataset_protocol={
            "collection_schema_version": COLLECTION_SCHEMA_VERSION,
            "pilot_version": PILOT_VERSION,
            "protocol_id": PROTOCOL_ID,
            "split_rule_id": SPLIT_RULE_ID,
            "heldout_fraction_per_task": HELDOUT_FRACTION,
            "observation_count": len(records),
            "trajectory_group_count": len(group_ids),
            "observations_per_group": len(TARGET_PROGRESS),
            "target_relative_progress": list(TARGET_PROGRESS),
            "accepted_groups_per_task": accepted_per_task,
        },
    )


def _validate_sample(
    sample: Any,
    *,
    task_id: int,
    state_id: int,
    manifest_parent: Path,
    observations_dir: Path,
    observation_type: Any,
) -> SourceRecord:
    if not isinstance(sample, dict):
        raise ExpandedDatasetError("sample record must be an object")
    sample_id = sample.get("sample_id")
    step_id = sample.get("step_id")
    relative_string = sample.get("observation_path")
    match = _SAMPLE_ID_PATTERN.fullmatch(sample_id or "")
    if (
        match is None
        or int(match.group("task")) != task_id
        or int(match.group("state")) != state_id
        or int(match.group("step")) != step_id
    ):
        raise ExpandedDatasetError(f"non-canonical sample ID: {sample_id!r}")
    relative = PurePosixPath(relative_string or "")
    expected = PurePosixPath("observations", f"{sample_id}.npz")
    if relative.is_absolute() or relative != expected:
        raise ExpandedDatasetError(f"non-canonical observation path: {relative}")
    resolved = (manifest_parent / Path(*relative.parts)).resolve(strict=True)
    if resolved.parent != observations_dir:
        raise ExpandedDatasetError("observation path escaped canonical directory")
    observation = observation_type.load(resolved)
    if sample.get("sha256") != _sha256_file(resolved):
        raise ExpandedDatasetError(f"observation hash differs: {sample_id}")
    if (
        observation.sample_id != sample_id
        or observation.task_id != str(task_id)
        or observation.initial_state_id != state_id
        or observation.episode_id != state_id
        or observation.step_id != step_id
        or observation.episode_success is not True
    ):
        raise ExpandedDatasetError(f"observation metadata differs: {sample_id}")
    image_hash = hashlib.sha256(
        np.ascontiguousarray(observation.base_rgb_raw).tobytes()
    ).hexdigest()
    return SourceRecord(
        sample_id=sample_id,
        source_observation_path=str(relative),
        resolved_observation_path=resolved,
        source_image_hash=f"sha256:{image_hash}",
    )


def _commit_group(records: Sequence[tuple[Any, Path]]) -> tuple[Path, ...]:
    temporary: list[Path] = []
    committed: list[Path] = []
    try:
        for record, final_path in records:
            if final_path.exists():
                raise ExpandedDatasetError(f"refusing to overwrite {final_path}")
            temp = final_path.with_name(f".{final_path.name}.tmp")
            temporary.append(temp)
            record.save(temp)
        for temp, (_, final_path) in zip(temporary, records, strict=True):
            os.replace(temp, final_path)
            committed.append(final_path)
    except BaseException:
        for path in (*temporary, *committed):
            path.unlink(missing_ok=True)
        raise
    return tuple(committed)


def _feasibility_report(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "task_id": row["task_id"],
            "available_initial_states": row["available_initial_states"],
            "attempted_states": row["attempted_states"],
            "successful_clean_openvla_trajectories": row[
                "successful_clean_openvla_trajectories"
            ],
            "failed_trajectories": row["failed_trajectories"],
            "accepted_unique_groups": row["accepted_unique_groups"],
            "rejected_states": row["failed_states"],
        }
        for row in manifest["task_results"]
    ]
    return {
        "schema_version": "pilot_v0_3_feasibility_report_v1",
        "status": "FEASIBILITY_AUDIT_COMPLETE",
        "split_readiness": manifest["split_readiness"],
        "maximum_available_unique_groups": manifest["feasibility"][
            "maximum_available_unique_groups"
        ],
        "actual_accepted_unique_groups": manifest["feasibility"][
            "actual_accepted_unique_groups"
        ],
        "per_task": rows,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
