"""Tests for Step 1 pair provenance and split consumer utilities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import torch

from scripts import step1_analyze_transfer as analysis
from scripts import step1_collect_heldout_pairs as collector
from scripts import step1_openvla_analysis as openvla_consumer
from scripts import step1_pi05_witness as pi05_consumer


def test_state_id_parser_supports_smoke_and_frozen_formal_range() -> None:
    assert collector._parse_state_ids("10") == (10,)
    assert collector._parse_state_ids("10-19") == tuple(range(10, 20))
    assert collector._parse_state_ids("10,12") == (10, 12)
    with pytest.raises(ValueError, match="state IDs"):
        collector._parse_state_ids("10,10")


def _capture(**changes):
    values = {
        "base_rgb": np.zeros((512, 512, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((512, 512, 3), dtype=np.uint8),
        "robot_state": np.arange(8, dtype=np.float64),
        "scene_state": np.arange(16, dtype=np.float64),
        "camera_state": np.arange(13, dtype=np.float64),
        "task_description": "pick up the bowl",
    }
    values.update(changes)
    return collector._Capture(**values)


def test_pair_scene_check_allows_texture_to_change_both_captured_camera_images() -> None:
    clean = _capture()
    changed_base = clean.base_rgb.copy()
    changed_base[0, 0, 0] = 1
    changed_wrist = clean.wrist_rgb.copy()
    changed_wrist[0, 0, 0] = 2
    collector._validate_scene_pair(
        clean,
        _capture(base_rgb=changed_base, wrist_rgb=changed_wrist),
    )

    changed_state = clean.scene_state.copy()
    changed_state[0] += 1
    with pytest.raises(RuntimeError, match="scene state"):
        collector._validate_scene_pair(clean, _capture(scene_state=changed_state))


class _ImageTools:
    def __init__(self) -> None:
        self.input = None

    def resize_with_pad(self, image, height, width):
        self.input = image.copy()
        assert (height, width) == (224, 224)
        return np.broadcast_to(image[0, 0], (224, 224, 3)).copy()

    @staticmethod
    def convert_to_uint8(image):
        return image.astype(np.uint8)


def test_pi05_adapter_applies_validated_rotation_before_resize() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    tools = _ImageTools()

    result = pi05_consumer._prepare_image(image, tools)

    np.testing.assert_array_equal(tools.input, image[::-1, ::-1])
    assert tools.input.flags.c_contiguous
    assert result.shape == (224, 224, 3)


def _write_pair_tree(root: Path) -> tuple[np.ndarray, np.ndarray]:
    state_dir = root / "state_10"
    state_dir.mkdir(parents=True)
    clean = np.zeros((512, 512, 3), dtype=np.uint8)
    adversarial = clean.copy()
    adversarial[0, 0] = (1, 2, 3)
    Image.fromarray(clean).save(state_dir / "clean.png")
    Image.fromarray(adversarial).save(state_dir / "adversarial.png")
    fixed_wrist = np.zeros((512, 512, 3), dtype=np.uint8)
    np.savez_compressed(
        state_dir / "fixed_inputs.npz",
        wrist_rgb_raw=fixed_wrist,
        robot_state=np.zeros(8, dtype=np.float64),
        scene_state=np.zeros(4, dtype=np.float64),
        camera_state=np.zeros(13, dtype=np.float64),
    )
    metadata = {
        "sample_id": "libero_spatial_task00_state10",
        "state_id": 10,
        "clean_rgb_sha256": collector.sha256_rgb(clean),
        "adv_rgb_sha256": collector.sha256_rgb(adversarial),
        "texture_sha256": "sha256:" + "a" * 64,
        "task_description": "pick up the bowl",
        "openvla_attacked_camera_field": "agentview_image",
        "pi05_corresponding_image_field": "base_0_rgb",
        "pi05_wrist_rgb_source": "clean_capture",
        "pi05_fixed_wrist_rgb_sha256": collector.sha256_rgb(fixed_wrist),
        "pi05_wrist_rgb_held_fixed": True,
        "no_policy_action_between_observations": True,
        "scene_state_identical": True,
        "camera_state_identical": True,
        "robot_state_identical": True,
    }
    (state_dir / "metadata.json").write_text(json.dumps(metadata))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "texture_sha256": metadata["texture_sha256"],
                "records": [{**metadata, "metadata_path": "state_10/metadata.json"}],
            }
        )
    )
    return clean, adversarial


def test_pi05_consumer_fails_closed_on_fixed_wrist_hash(tmp_path) -> None:
    pairs = tmp_path / "pairs"
    _write_pair_tree(pairs)
    metadata_path = pairs / "state_10/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["pi05_fixed_wrist_rgb_sha256"] = "sha256:" + "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="fixed wrist RGB identity mismatch"):
        pi05_consumer._load_pairs(pairs)


def test_both_consumers_fail_closed_on_the_same_pair_hashes(tmp_path) -> None:
    pairs = tmp_path / "pairs"
    _, adversarial = _write_pair_tree(pairs)

    _, openvla_records = openvla_consumer._load_pairs(pairs)
    _, pi05_records = pi05_consumer._load_pairs(pairs)

    assert openvla_records[0]["sample_id"] == pi05_records[0]["sample_id"]
    assert (
        openvla_records[0]["metadata"]["adv_rgb_sha256"]
        == pi05_records[0]["metadata"]["adv_rgb_sha256"]
    )

    adversarial[0, 0, 0] = 9
    Image.fromarray(adversarial).save(pairs / "state_10/adversarial.png")
    with pytest.raises(ValueError, match="identity mismatch"):
        openvla_consumer._load_pairs(pairs)
    with pytest.raises(ValueError, match="identity mismatch"):
        pi05_consumer._load_pairs(pairs)


def test_spearman_helper_reports_rho_and_p_value() -> None:
    rho, p_value = analysis._correlation(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        "test",
    )
    assert rho == pytest.approx(-1.0)
    assert p_value == pytest.approx(0.0)


def test_transfer_analysis_requires_frozen_pytorch_witness_contract() -> None:
    summary = {
        "backend": "pytorch",
        "inference_contract": {
            "model_eval": True,
            "torch_no_grad_guard": True,
            "model_parameters_require_grad": False,
            "official_embed_prefix_slice_equal": True,
        },
    }
    analysis._validate_pi05_contract(summary)

    summary["backend"] = "jax_nnx"
    with pytest.raises(ValueError, match="frozen PyTorch contract"):
        analysis._validate_pi05_contract(summary)


def test_pi05_policy_input_changes_only_base_camera() -> None:
    tools = _ImageTools()
    fixed = {
        "wrist_rgb": np.ones((4, 5, 3), dtype=np.uint8),
        "robot_state": np.arange(8, dtype=np.float64),
        "prompt": "pick up the bowl",
        "image_tools": tools,
    }
    clean = pi05_consumer._policy_input(
        base_rgb=np.zeros((4, 5, 3), dtype=np.uint8), **fixed
    )
    adversarial = pi05_consumer._policy_input(
        base_rgb=np.ones((4, 5, 3), dtype=np.uint8), **fixed
    )

    assert clean["observation/state"] is not adversarial["observation/state"]
    np.testing.assert_array_equal(
        clean["observation/state"], adversarial["observation/state"]
    )
    assert clean["prompt"] == adversarial["prompt"]
    assert not np.array_equal(
        clean["observation/image"], adversarial["observation/image"]
    )
    np.testing.assert_array_equal(
        clean["observation/wrist_image"],
        adversarial["observation/wrist_image"],
    )


class _FakeTorchImageEncoder:
    def __init__(self) -> None:
        self.grad_enabled: list[bool] = []

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        scalar = image.float().mean(dim=tuple(range(1, image.ndim)), keepdim=False)
        return scalar[:, None, None].expand(-1, 256, 2048).to(torch.bfloat16)


class _FakeTorchPi05(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.paligemma_with_expert = _FakeTorchImageEncoder()
        self.preprocess_train_values: list[bool] = []

    def _preprocess_observation(self, observation, *, train):
        self.preprocess_train_values.append(train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, image_masks, lang_tokens, lang_masks):
        del image_masks, lang_masks
        image_tokens = [
            self.paligemma_with_expert.embed_image(image) for image in images
        ]
        language = torch.zeros(
            lang_tokens.shape[0],
            lang_tokens.shape[1],
            2048,
            dtype=image_tokens[0].dtype,
        )
        prefix = torch.cat([*image_tokens, language], dim=1)
        return prefix, None, None


class _FakeTorchPolicy:
    @staticmethod
    def _input_transform(observation):
        base = observation["observation/image"].astype(np.float32)
        wrist = observation["observation/wrist_image"].astype(np.float32)
        padded = np.zeros_like(wrist)
        images = {
            "base_0_rgb": base,
            "left_wrist_0_rgb": wrist,
            "right_wrist_0_rgb": padded,
        }
        return {
            "images": images,
            "image_masks": {
                key: np.asarray(True, dtype=np.bool_) for key in images
            },
            "state": observation["observation/state"].astype(np.float32),
            "tokenized_prompt": np.zeros(4, dtype=np.int64),
            "tokenized_prompt_mask": np.ones(4, dtype=np.bool_),
        }


class _FakeObservation:
    @staticmethod
    def from_dict(inputs):
        return SimpleNamespace(**inputs)


def test_torch_pi05_p2_matches_official_prefix_slice_under_no_grad() -> None:
    model = _FakeTorchPi05()
    runtime = SimpleNamespace(
        torch=torch,
        policy=_FakeTorchPolicy(),
        observation_type=_FakeObservation,
        model=model,
        device=torch.device("cpu"),
    )
    observation = {
        "observation/image": np.ones((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image": np.full((2, 2, 3), 2, dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the bowl",
    }

    p2, image_p2 = pi05_consumer._extract_p2(runtime, observation)

    assert p2.shape == (256, 2048)
    assert tuple(image_p2) == pi05_consumer.IMAGE_KEYS
    assert model.training is False
    assert model.preprocess_train_values == [False]
    assert model.paligemma_with_expert.grad_enabled
    assert not any(model.paligemma_with_expert.grad_enabled)
