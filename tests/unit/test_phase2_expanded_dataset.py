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
    HELDOUT_FRACTION,
    PILOT_VERSION,
    PROTOCOL_ID,
    SPLIT_RULE_ID,
    TARGET_PROGRESS,
    capacity_report,
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
