"""Step 1B gates, update semantics, passive snapshots, and completion tests."""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
import random
import subprocess
import sys
from typing import Optional, Union

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openvla/experiments/robot"))
import step1b_trajectory as trajectory  # noqa: E402

ENTRYPOINT = ROOT / "openvla/experiments/robot/libero/attack_openvla.py"
BASE = "e8d7d5eb97d08cbad50d1f8295651867c43d9be6"


def _entrypoint_definitions():
    tree = ast.parse(ENTRYPOINT.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ("GenerateConfig", "_validate_step1_formal_config")]
    namespace = dict(dataclass=dataclass, Optional=Optional, Union=Union,
                     Path=Path, LEGACY_OBJECTIVE="legacy",
                     O2_DISPLACEMENT_OBJECTIVE="o2_displacement")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENTRYPOINT), "exec"), namespace)
    return namespace["GenerateConfig"], namespace["_validate_step1_formal_config"]


def _config(root):
    return {**trajectory.FROZEN_CONFIG,
            "step1_output_dir": str(root), "source_git_commit": "a" * 40,
            "train_state_ids": list(range(10)), "heldout_state_ids": list(range(10, 20)),
            "renderer_epsilon": 128 / 255, "renderer_position_offset": [0., 0., 0.]}


def test_historical_formal_and_renderer_implementations_unchanged():
    before = subprocess.check_output(
        ["git", "show", f"{BASE}:{ENTRYPOINT.relative_to(ROOT)}"], cwd=ROOT, text=True)
    old, new = ast.parse(before), ast.parse(ENTRYPOINT.read_text())
    for name in ("_validate_step1_formal_config", "DifferentiableRenderer"):
        a = next(n for n in old.body if getattr(n, "name", None) == name)
        b = next(n for n in new.body if getattr(n, "name", None) == name)
        assert ast.dump(a) == ast.dump(b)
    Config, validate = _entrypoint_definitions()
    cfg = Config(attack_objective="o2_displacement", task_id=0,
                 num_trials_per_task=0, live_test_enabled=False,
                 step1_formal=True, step1_output_dir="/tmp/example")
    validate(cfg)
    assert cfg.step1b_mature_trajectory is False
    cfg.attack_iters = 5000
    with pytest.raises(ValueError, match="frozen"):
        validate(cfg)


def test_step1b_gate_accepts_only_independent_frozen_mode(tmp_path):
    config = _config(tmp_path)
    trajectory.validate_step1b_config(config)
    with pytest.raises(ValueError, match="frozen"):
        trajectory.validate_step1b_config({**config, "step1_formal": True})
    with pytest.raises(ValueError, match="frozen"):
        trajectory.validate_step1b_config({**config, "step1_output_dir": None})
    # Every other frozen value is independently exercised, not just iterations.
    for key in trajectory.FROZEN_CONFIG:
        if key == "step1b_mature_trajectory":
            continue
        with pytest.raises(ValueError, match="frozen"):
            trajectory.validate_step1b_config({**config, key: "changed"})


class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.full((3, 3), 0.01))
        self.register_buffer("calibration", torch.tensor(0.8))
        self.epsilon = 128 / 255
        self.bake_calls = 0

    def get_texture_param(self):
        return self.parameter

    def get_baked_adv_texture(self):
        assert not torch.is_grad_enabled()
        self.bake_calls += 1
        return (0.5 + self.epsilon * torch.tanh(self.parameter)).reshape(1, 1, 3, 3)


def test_schedule_post_update_and_snapshot_does_not_change_training(tmp_path):
    assert trajectory.CHECKPOINT_SCHEDULE == (10, 100, 500, 1000, 2000, 5000)
    assert sorted(set(trajectory.CHECKPOINT_SCHEDULE)) == list(trajectory.CHECKPOINT_SCHEDULE)
    assert trajectory.checkpoint_iteration(9) == 10
    assert trajectory.checkpoint_iteration(10) is None
    training = tmp_path / "training"
    training.mkdir()
    trajectory.initialize_manifest(training, _config(tmp_path))
    renderer = Renderer()
    optimizer = torch.optim.SGD(renderer.parameters(), lr=0.05, momentum=0.9)
    # One update precedes the snapshot, just as at the real call site.
    renderer.parameter.sum().backward()
    optimizer.step()
    before = {k: v.clone() for k, v in renderer.state_dict().items()}
    grad = renderer.parameter.grad.clone()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    torch_rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    metrics = {"total_loss": -0.4, "o2_displacement": 0.4}
    trajectory.save_checkpoint(renderer, training, 9, metrics)
    trajectory.save_checkpoint(renderer, training, 10, metrics)
    assert renderer.bake_calls == 1
    for k, v in before.items():
        assert torch.equal(renderer.state_dict()[k], v)
    assert torch.equal(renderer.parameter.grad, grad)
    assert optimizer.state_dict()["param_groups"] == optimizer_state["param_groups"]
    assert torch.equal(optimizer.state_dict()["state"][0]["momentum_buffer"],
                       optimizer_state["state"][0]["momentum_buffer"])
    assert renderer.training and renderer.parameter.requires_grad and torch.is_grad_enabled()
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    now = np.random.get_state()
    assert now[0] == numpy_rng[0] and np.array_equal(now[1], numpy_rng[1]) and now[2:] == numpy_rng[2:]
    target = training / "checkpoints/iter_000010"
    assert torch.equal(torch.load(target / "parameter.pt", weights_only=True), renderer.parameter)
    meta = trajectory.read_json(target / "metadata.json")
    assert meta["iteration"] == 10 and meta["pre_update_total_loss"] == -0.4
    assert meta["pre_update_o2_displacement"] == 0.4
    assert meta["texture_sha256"] == trajectory.sha256_file(target / "attack_texture.png")
    with pytest.raises(ValueError, match="schedule"):
        trajectory.save_checkpoint(renderer, training, 9, metrics)


def test_checkpoint_hook_follows_actual_update_and_is_opt_in():
    source = ENTRYPOINT.read_text()
    update = source.index("renderer.adv_noise.data -= pgd_step * grad.sign()")
    hook = source.index("save_checkpoint(renderer, Path(save_dir), i, step_metrics)")
    assert update < hook
    assert "if cfg.step1b_mature_trajectory:" in source[update:hook]
    assert source.index('if restoration_failure is not None:') < source.index('finalize_trajectory(step1_run_dir)')


@pytest.fixture
def completed_run(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    config = _config(tmp_path)
    trajectory.write_json(tmp_path / "config.json", config)
    trajectory.initialize_manifest(training, config)
    renderer = Renderer()
    metrics = []
    for i in range(5000):
        with torch.no_grad():
            renderer.parameter.add_(0.00001)
        row = dict(iteration=i, total_loss=-0.4, o2_displacement=0.4,
                   action_loss=0., feature_loss=0., attack_objective="o2_displacement",
                   texture_gradient_norm=1., image_gradient_norm_max=1.,
                   parameter_change_linf=0.00001,
                   maximum_texture_perturbation=float((torch.tanh(renderer.parameter.detach()) * renderer.epsilon).abs().max()))
        metrics.append(row)
        trajectory.save_checkpoint(renderer, training, i, row)
    torch.save(renderer.parameter.detach(), training / "parameter.pt")
    with torch.no_grad():
        Image.fromarray((renderer.get_baked_adv_texture().squeeze(0).numpy() * 255).astype(np.uint8)).save(training / "final_attack_texture.png")
    texture_hash = trajectory.sha256_file(training / "final_attack_texture.png")
    (training / "texture_sha256.txt").write_text(texture_hash + "\n")
    trajectory.write_json(training / "training_summary.json", {
        "texture_sha256": texture_hash, "num_iterations": 5000,
        "num_training_frames": 10, "pi05_loaded_during_training": False,
        "texture_budget_respected": True,
    })
    import json
    (training / "Ep0_step_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics))
    np.save(training / "loss_history.npy", np.array([r["total_loss"] for r in metrics]))
    trajectory.write_json(tmp_path / "asset_restoration.json", {k: True for k in (
        "pass", "xml_restored", "clean_texture_restored", "xml_backup_removed", "texture_backup_removed")})
    return tmp_path


def test_manifest_completion_and_final_equivalence(completed_run):
    with pytest.raises(ValueError, match="not complete"):
        trajectory.audit_checkpoints(completed_run)
    trajectory.finalize_trajectory(completed_run)
    manifest = trajectory.audit_checkpoints(completed_run)
    assert manifest["completed_checkpoints"] == list(trajectory.CHECKPOINT_SCHEDULE)
    assert manifest["final_parameter_equal"] and manifest["final_texture_sha256_equal"]


@pytest.mark.parametrize("artifact", ["parameter", "texture", "restoration", "hash", "metrics"])
def test_completion_fails_closed_on_invalid_artifacts(completed_run, artifact):
    training = completed_run / "training"
    if artifact == "parameter":
        torch.save(torch.zeros(3, 3), training / "parameter.pt")
    elif artifact == "texture":
        Image.new("RGB", (3, 1)).save(training / "final_attack_texture.png")
    elif artifact == "restoration":
        trajectory.write_json(completed_run / "asset_restoration.json", {"pass": False})
    elif artifact == "hash":
        (training / "checkpoints/iter_000100/parameter.pt").write_bytes(b"corrupt")
    else:
        (training / "Ep0_step_metrics.jsonl").write_text("")
    with pytest.raises(ValueError):
        trajectory.finalize_trajectory(completed_run)
    manifest = trajectory.read_json(training / "checkpoints/checkpoint_manifest.json")
    assert manifest["training_complete"] is False


def test_analysis_refuses_incomplete_training_before_loading_pairs(completed_run, monkeypatch):
    from scripts import step1b_analyze_trajectory as analysis
    monkeypatch.setattr(analysis.source, "_load_pairs", lambda *_: pytest.fail("loaded heldout early"))
    with pytest.raises(ValueError, match="not complete"):
        analysis.analyze_checkpoint(completed_run, 10)


def test_posthoc_reuses_existing_statistics_with_trajectory_provenance(completed_run, monkeypatch):
    from scripts import step1b_analyze_trajectory as analysis
    trajectory.finalize_trajectory(completed_run)
    directory = completed_run / "trajectory/iter_000010"
    (directory / "analysis").mkdir(parents=True)
    calls = []
    def validate(root, iteration, manifest):
        assert manifest["training_complete"] is True
        assert root == completed_run and iteration == 10
        calls.append("validated")
        return directory
    def existing_analyzer(args):
        assert calls == ["validated"]
        assert args.training_dir is None  # Never substitute the final texture.
        assert args.formal is False       # Preserve the old 10-iter formal mode.
        assert args.pairs_dir == directory / "heldout_pairs"
        calls.append("analyzed")
        return {"state_spearman_rho": 0.2}
    monkeypatch.setattr(analysis, "validate_inputs", validate)
    monkeypatch.setattr(analysis.transfer, "_run", existing_analyzer)
    result = analysis.analyze_checkpoint(completed_run, 10)
    assert calls == ["validated", "analyzed"]
    assert result["checkpoint_iteration"] == 10
    assert result["training_total_iterations"] == 5000
    assert result["all_checkpoints_frozen_before_analysis"] is True


def test_posthoc_rejects_texture_hash_mismatch_before_statistics(completed_run, monkeypatch):
    from scripts import step1b_analyze_trajectory as analysis
    trajectory.finalize_trajectory(completed_run)
    pairs = {"heldout_state_ids": list(range(10, 20)), "num_pairs": 10,
             "texture_sha256": "sha256:wrong", "xml_restored": True,
             "clean_texture_unchanged": True}
    records = [{"state_id": i} for i in range(10, 20)]
    monkeypatch.setattr(analysis.source, "_load_pairs", lambda *_: (pairs, records))
    monkeypatch.setattr(analysis.witness, "_load_pairs", lambda *_: (pairs, records))
    monkeypatch.setattr(analysis.transfer, "_run", lambda *_: pytest.fail("statistics before identity gate"))
    with pytest.raises(ValueError, match="pair provenance"):
        analysis.analyze_checkpoint(completed_run, 10)


def test_trajectory_summary_requires_all_six_and_does_not_run_models(completed_run, monkeypatch):
    from scripts import step1b_analyze_trajectory as analysis
    trajectory.finalize_trajectory(completed_run)
    manifest = trajectory.audit_checkpoints(completed_run)
    monkeypatch.setattr(analysis, "validate_inputs",
                        lambda root, i, _: root / "trajectory" / f"iter_{i:06d}")
    monkeypatch.setattr(analysis.transfer, "_run", lambda *_: pytest.fail("summary reran statistics"))
    for record in manifest["checkpoints"]:
        iteration = record["iteration"]
        directory = completed_run / "trajectory" / f"iter_{iteration:06d}" / "analysis"
        directory.mkdir(parents=True)
        summary = dict(checkpoint_iteration=iteration, training_total_iterations=5000,
                       all_checkpoints_frozen_before_analysis=True,
                       training_source_git_commit=manifest["source_git_commit"],
                       texture_sha256=record["texture_sha256"], num_states=10,
                       heldout_state_ids=list(range(10,20)),
                       state_displacement_o2=list(range(10)), state_displacement_p2=list(range(10,20)))
        for key in ("state_spearman_rho", "state_spearman_p", "token_spearman_mean",
                    "token_spearman_median", "token_spearman_min", "token_spearman_max"):
            summary[key] = 0.2
        trajectory.write_json(directory / "summary.json", summary)
        for name in ("displacement_summary.csv", "token_spearman.csv"):
            (directory / name).write_text("fixture\n")
    result = analysis.summarize(completed_run)
    assert [r["iteration"] for r in result["checkpoints"]] == list(trajectory.CHECKPOINT_SCHEDULE)
    assert result["checkpoints"][0]["mean_d_O2"] == 4.5
    assert result["checkpoints"][0]["median_d_P2"] == 14.5
    with pytest.raises(FileExistsError, match="fresh"):
        analysis.summarize(completed_run)
