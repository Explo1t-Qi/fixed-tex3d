from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROBOT_ROOT = Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
sys.path.insert(0, str(ROBOT_ROOT))

from phase2_expanded_dataset import (  # noqa: E402
    CHECKPOINT_IDENTITY,
    COLLECTION_SCHEMA_VERSION,
    ExpandedDatasetError,
    HELDOUT_FRACTION,
    PILOT_VERSION,
    PROTOCOL_ID,
    SPLIT_RULE_ID,
    TARGET_PROGRESS,
    capacity_report,
    collect_expanded_observations,
    sample_indices,
    validate_expanded_collection,
)


class _Suite:
    def get_num_tasks(self) -> int:
        return 10

    def get_task_init_states(self, task_id: int) -> list[int]:
        return list(range(50 - task_id % 2))


class _Observation:
    records: dict[Path, SimpleNamespace] = {}

    @classmethod
    def load(cls, path: Path) -> SimpleNamespace:
        return cls.records[path]


def test_capacity_audit_uses_actual_suite_lengths() -> None:
    report = capacity_report(_Suite(), libero_revision="fixture-revision")
    assert report["status"] == "CAPACITY_AUDIT_COMPLETE"
    assert report["available_initial_states_per_task"]["0"] == 50
    assert report["available_initial_states_per_task"]["1"] == 49
    assert report["maximum_unique_groups"] == 495


def test_six_progress_sampling_is_unique_or_rejected_explicitly() -> None:
    assert len(set(sample_indices(100))) == len(TARGET_PROGRESS)
    assert TARGET_PROGRESS == (0.10, 0.25, 0.40, 0.55, 0.70, 0.90)
    assert len(set(sample_indices(6))) < len(TARGET_PROGRESS)


def _expanded_fixture(tmp_path: Path) -> Path:
    observations = tmp_path / "observations"
    observations.mkdir()
    task_results = []
    accepted_counts = {}
    _Observation.records = {}
    for task_id in range(10):
        group_count = 2 + task_id % 2
        groups = []
        for state_id in range(group_count):
            samples = []
            for progress_index, progress in enumerate(TARGET_PROGRESS):
                step_id = 10 + progress_index
                sample_id = (
                    f"libero_spatial__task{task_id:02d}__state{state_id:02d}"
                    f"__step{step_id:04d}"
                )
                path = observations / f"{sample_id}.npz"
                path.write_bytes(b"fixture")
                _Observation.records[path.resolve()] = SimpleNamespace(
                    sample_id=sample_id,
                    task_id=str(task_id),
                    initial_state_id=state_id,
                    episode_id=state_id,
                    step_id=step_id,
                    episode_success=True,
                    base_rgb_raw=np.full((2, 2, 3), task_id + state_id, dtype=np.uint8),
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "step_id": step_id,
                        "target_relative_progress": progress,
                        "actual_normalized_episode_progress": step_id / 100,
                        "observation_path": f"observations/{sample_id}.npz",
                        "sha256": "sha256:"
                        + __import__("hashlib").sha256(b"fixture").hexdigest(),
                    }
                )
            groups.append(
                {
                    "task_id": task_id,
                    "initial_state_id": state_id,
                    "trajectory_length": 101,
                    "episode_success": True,
                    "samples": samples,
                }
            )
        accepted_counts[str(task_id)] = group_count
        task_results.append(
            {
                "task_id": task_id,
                "task_language": f"task {task_id}",
                "available_initial_states": group_count,
                "attempted_states": group_count,
                "successful_clean_openvla_trajectories": group_count,
                "failed_trajectories": 0,
                "accepted_unique_groups": group_count,
                "attempted_state_ids": list(range(group_count)),
                "accepted_state_ids": list(range(group_count)),
                "failed_states": [],
                "accepted_groups": groups,
            }
        )
    total_groups = sum(accepted_counts.values())
    manifest = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "pilot_version": PILOT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "suite": "libero_spatial",
        "run_status": "COMPLETED",
        "split_readiness": "READY_FOR_SPLIT",
        "provenance": {
            "code_commit": "a" * 40,
            "shared_feature_commit": "b" * 40,
            "libero_revision": "fixture",
        },
        "rollout": {"checkpoint_identity": CHECKPOINT_IDENTITY},
        "protocol": {
            "task_ids": list(range(10)),
            "observations_per_group": 6,
            "target_relative_progress": list(TARGET_PROGRESS),
            "group_identity_fields": ["task_id", "initial_state_id"],
            "split_rule_id": SPLIT_RULE_ID,
            "heldout_fraction_per_task": HELDOUT_FRACTION,
        },
        "runtime": {
            "libero_revision": "fixture",
            "discovered_task_count": 10,
            "task_ids": list(range(10)),
        },
        "coverage": {
            "actual_total_groups": total_groups,
            "actual_total_observations": total_groups * 6,
            "accepted_groups_per_task": accepted_counts,
        },
        "task_results": task_results,
    }
    path = tmp_path / "collection_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    feasibility = {
        "schema_version": "pilot_v0_3_feasibility_report_v1",
        "status": "FEASIBILITY_AUDIT_COMPLETE",
        "split_readiness": "READY_FOR_SPLIT",
        "maximum_available_unique_groups": total_groups,
        "actual_accepted_unique_groups": total_groups,
        "per_task": [
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
            for row in task_results
        ],
    }
    manifest["feasibility"] = {
        "maximum_available_unique_groups": total_groups,
        "actual_accepted_unique_groups": total_groups,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "feasibility_report.json").write_text(
        json.dumps(feasibility), encoding="utf-8"
    )
    return path


def test_expanded_manifest_accepts_variable_groups_and_serializes_protocol(
    tmp_path: Path,
) -> None:
    source = validate_expanded_collection(
        _expanded_fixture(tmp_path), observation_type=_Observation
    )
    assert len(source.records) == 25 * 6
    assert source.dataset_protocol["trajectory_group_count"] == 25
    assert source.dataset_protocol["accepted_groups_per_task"]["1"] == 3
    assert len({record.sample_id for record in source.records}) == len(source.records)


def test_expanded_manifest_rejects_duplicate_task_state_progress_identity(
    tmp_path: Path,
) -> None:
    path = _expanded_fixture(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest["task_results"][0]["accepted_groups"][0]["samples"]
    samples[1]["target_relative_progress"] = samples[0]["target_relative_progress"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="progress set differs"):
        validate_expanded_collection(path, observation_type=_Observation)


class _CollectionSuite:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def get_num_tasks(self) -> int:
        return 10

    def get_task_init_states(self, task_id: int) -> list[tuple[int, int]]:
        return [(task_id, state_id) for state_id in range(self.counts[task_id])]

    def get_task(self, task_id: int) -> SimpleNamespace:
        return SimpleNamespace(language=f"task {task_id}")


class _CollectionObservation:
    save_calls = 0
    fail_on_save_call: int | None = None

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def save(self, path: Path) -> None:
        type(self).save_calls += 1
        with path.open("wb") as stream:
            np.savez(
                stream,
                sample_id=np.array(self.sample_id),
                task_id=np.array(self.task_id),
                initial_state_id=np.array(self.initial_state_id),
                episode_id=np.array(self.episode_id),
                step_id=np.array(self.step_id),
                episode_success=np.array(self.episode_success),
                base_rgb_raw=self.base_rgb_raw,
            )
        if type(self).save_calls == type(self).fail_on_save_call:
            raise RuntimeError("injected partial group write")

    @classmethod
    def load(cls, path: Path) -> SimpleNamespace:
        with np.load(path, allow_pickle=False) as archive:
            return SimpleNamespace(
                sample_id=str(archive["sample_id"]),
                task_id=str(archive["task_id"]),
                initial_state_id=int(archive["initial_state_id"]),
                episode_id=int(archive["episode_id"]),
                step_id=int(archive["step_id"]),
                episode_success=bool(archive["episode_success"]),
                base_rgb_raw=archive["base_rgb_raw"].copy(),
            )


class _FakeCollector:
    PILOT_SUITE = "libero_spatial"
    _CAMERA_RESOLUTION = 256
    _NUM_DUMMY_STEPS = 10
    _MAX_POLICY_ACTIONS = 520

    class _EpisodeCollectionError(RuntimeError):
        category = "fixture"

    def __init__(self, *, interrupt_at: tuple[int, int] | None = None) -> None:
        self.interrupt_at = interrupt_at
        self.calls: list[tuple[int, int]] = []

    @staticmethod
    def _OpenVLAActionConfig(**values: object) -> SimpleNamespace:
        return SimpleNamespace(**values)

    def _collect_episode(self, **values: object) -> tuple[list[SimpleNamespace], bool]:
        pair = (int(values["task_id"]), int(values["initial_state_id"]))
        self.calls.append(pair)
        if pair == self.interrupt_at:
            raise KeyboardInterrupt
        trajectory = [
            SimpleNamespace(
                step_id=step_id,
                base_rgb_raw=np.full((2, 2, 3), sum(pair), dtype=np.uint8),
                wrist_rgb_raw=np.full((2, 2, 3), step_id, dtype=np.uint8),
                state=np.array([*pair, step_id], dtype=np.float32),
            )
            for step_id in range(20)
        ]
        return trajectory, True


class _FinalValidationFailObservation(_CollectionObservation):
    @classmethod
    def load(cls, path: Path) -> SimpleNamespace:
        del path
        raise RuntimeError("injected final validation failure")


class _ProgressRecorder:
    def __init__(self, *, total: int, initial: int = 0) -> None:
        self.total = total
        self.n = initial
        self.messages: list[str] = []
        self.postfixes: list[dict[str, object]] = []

    def set_postfix(self, values: dict[str, object], *, refresh: bool = False) -> None:
        del refresh
        self.postfixes.append(dict(values))

    def update(self, count: int = 1) -> None:
        self.n += count

    def write(self, message: str) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass


def _run_fixture_collection(
    *,
    root: Path,
    checkpoint: Path,
    suite: _CollectionSuite,
    collector: _FakeCollector,
    resume: bool = False,
    code_commit: str = "a" * 40,
    libero_revision: str = "fixture-revision",
    progress_instances: list[_ProgressRecorder] | None = None,
) -> object:
    def progress_factory(*, total: int, initial: int) -> _ProgressRecorder:
        progress = _ProgressRecorder(total=total, initial=initial)
        if progress_instances is not None:
            progress_instances.append(progress)
        return progress

    return collect_expanded_observations(
        model=object(),
        processor=object(),
        pretrained_checkpoint=checkpoint,
        libero_revision=libero_revision,
        code_commit=code_commit,
        shared_feature_commit="b" * 40,
        output_dir=root,
        resume=resume,
        progress_factory=progress_factory,
        _collector=collector,
        _runtime=object(),
        _suite=suite,
        _observation_type=_CollectionObservation,
    )


def test_collection_progress_total_uses_actual_suite_capacity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    progress_instances: list[_ProgressRecorder] = []
    counts = [2, *([1] * 9)]
    _CollectionObservation.save_calls = 0
    _CollectionObservation.fail_on_save_call = None

    result = _run_fixture_collection(
        root=tmp_path / "collection",
        checkpoint=checkpoint,
        suite=_CollectionSuite(counts),
        collector=_FakeCollector(),
        progress_instances=progress_instances,
    )

    assert result.status == "EXPANDED_COLLECTION_COMPLETE"
    assert progress_instances[0].total == sum(counts) == 11
    assert progress_instances[0].n == sum(counts)
    assert set(progress_instances[0].postfixes[-1]) == {
        "task",
        "state",
        "accepted",
        "policy_failed",
        "sampling_rejected",
    }


def test_fresh_collection_rejects_nonempty_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "collection"
    output.mkdir()
    (output / "existing").write_text("fixture", encoding="utf-8")

    with pytest.raises(ExpandedDatasetError, match="must be fresh"):
        _run_fixture_collection(
            root=output,
            checkpoint=checkpoint,
            suite=_CollectionSuite([1] * 10),
            collector=_FakeCollector(),
        )


def _make_interrupted_collection(
    tmp_path: Path,
) -> tuple[Path, Path, _CollectionSuite]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "collection"
    suite = _CollectionSuite([2, *([1] * 9)])
    _CollectionObservation.save_calls = 0
    _CollectionObservation.fail_on_save_call = None
    with pytest.raises(KeyboardInterrupt):
        _run_fixture_collection(
            root=output,
            checkpoint=checkpoint,
            suite=suite,
            collector=_FakeCollector(interrupt_at=(0, 1)),
        )
    progress = json.loads(
        (output / "collection_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "INTERRUPTED_RESUMABLE"
    assert progress["counts"]["completed"] == 1
    assert not (output / "collection_manifest.json").exists()
    return checkpoint, output, suite


def test_resume_skips_verified_state_without_duplicates_and_matches_clean_run(
    tmp_path: Path,
) -> None:
    checkpoint, output, suite = _make_interrupted_collection(tmp_path)
    resumed_collector = _FakeCollector()
    resumed = _run_fixture_collection(
        root=output,
        checkpoint=checkpoint,
        suite=suite,
        collector=resumed_collector,
        resume=True,
    )
    assert (0, 0) not in resumed_collector.calls

    clean = _run_fixture_collection(
        root=tmp_path / "clean",
        checkpoint=checkpoint,
        suite=suite,
        collector=_FakeCollector(),
    )
    resumed_manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    clean_manifest = json.loads(clean.manifest_path.read_text(encoding="utf-8"))
    for key in (
        "protocol",
        "provenance",
        "rollout",
        "runtime",
        "feasibility",
        "coverage",
        "task_results",
    ):
        assert resumed_manifest[key] == clean_manifest[key]
    sample_ids = [
        sample["sample_id"]
        for row in resumed_manifest["task_results"]
        for group in row["accepted_groups"]
        for sample in group["samples"]
    ]
    assert len(sample_ids) == len(set(sample_ids)) == len(resumed.sample_paths)
    assert resumed_manifest["execution"] == {
        "completed_state_count": 11,
        "progress_schema_version": "pilot_v0_3_collection_progress_v1",
        "resume_count": 1,
        "resumed": True,
    }


def test_resume_rejects_corrupted_observation_hash(tmp_path: Path) -> None:
    checkpoint, output, suite = _make_interrupted_collection(tmp_path)
    observation = next((output / "observations").glob("*.npz"))
    with observation.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ExpandedDatasetError, match="hash differs"):
        _run_fixture_collection(
            root=output,
            checkpoint=checkpoint,
            suite=suite,
            collector=_FakeCollector(),
            resume=True,
        )


@pytest.mark.parametrize("mismatch", ["protocol", "commit", "revision", "checkpoint"])
def test_resume_rejects_identity_mismatch(tmp_path: Path, mismatch: str) -> None:
    checkpoint, output, suite = _make_interrupted_collection(tmp_path)
    code_commit = "a" * 40
    libero_revision = "fixture-revision"
    if mismatch == "protocol":
        progress_path = output / "collection_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["protocol_id"] = "different-protocol"
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
    elif mismatch == "commit":
        code_commit = "c" * 40
    elif mismatch == "revision":
        libero_revision = "different-revision"
    else:
        checkpoint = tmp_path / "different-checkpoint"
        checkpoint.mkdir()

    with pytest.raises(ExpandedDatasetError, match="resume identity mismatch"):
        _run_fixture_collection(
            root=output,
            checkpoint=checkpoint,
            suite=suite,
            collector=_FakeCollector(),
            resume=True,
            code_commit=code_commit,
            libero_revision=libero_revision,
        )


def test_partial_group_is_not_completed_and_is_rerun_on_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "collection"
    suite = _CollectionSuite([1] * 10)
    _CollectionObservation.save_calls = 0
    _CollectionObservation.fail_on_save_call = 3

    with pytest.raises(RuntimeError, match="partial group"):
        _run_fixture_collection(
            root=output,
            checkpoint=checkpoint,
            suite=suite,
            collector=_FakeCollector(),
        )
    progress = json.loads(
        (output / "collection_progress.json").read_text(encoding="utf-8")
    )
    assert progress["counts"]["completed"] == 0
    assert list((output / "observations").iterdir()) == []

    _CollectionObservation.save_calls = 0
    _CollectionObservation.fail_on_save_call = None
    resumed_collector = _FakeCollector()
    result = _run_fixture_collection(
        root=output,
        checkpoint=checkpoint,
        suite=suite,
        collector=resumed_collector,
        resume=True,
    )
    assert resumed_collector.calls[0] == (0, 0)
    assert result.status == "EXPANDED_COLLECTION_COMPLETE"


def test_final_validation_failure_does_not_leave_completed_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "collection"
    _FinalValidationFailObservation.save_calls = 0
    _FinalValidationFailObservation.fail_on_save_call = None

    with pytest.raises(RuntimeError, match="final validation failure"):
        collect_expanded_observations(
            model=object(),
            processor=object(),
            pretrained_checkpoint=checkpoint,
            libero_revision="fixture-revision",
            code_commit="a" * 40,
            shared_feature_commit="b" * 40,
            output_dir=output,
            progress_factory=lambda **values: _ProgressRecorder(**values),
            _collector=_FakeCollector(),
            _runtime=object(),
            _suite=_CollectionSuite([1] * 10),
            _observation_type=_FinalValidationFailObservation,
        )

    assert not (output / "collection_manifest.json").exists()
    assert not (output / "feasibility_report.json").exists()
    progress = json.loads(
        (output / "collection_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "INTERRUPTED_RESUMABLE"
    assert progress["counts"]["completed"] == 10
