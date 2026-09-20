"""Frozen extraction contracts for action-predictive representation probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


ACTION_DIM = 7
NUM_VISUAL_TOKENS = 256
PI05_P2_WIDTH = 2048
PI05_P2_DEFINITION_ID = "pi05_p2_embed_image_no_manual_scaling_v2"


class ActionRepresentationError(RuntimeError):
    """Raised when a representation/action extraction contract is violated."""


@dataclass(frozen=True)
class DeepNodeIdentity:
    module_path: str
    total_layers: int
    zero_based_layer_index: int
    one_based_layer_index: int
    capture_point: str
    visual_token_slice: str


@dataclass(frozen=True)
class ModelActionRepresentation:
    projected: torch.Tensor
    deep: torch.Tensor
    deployed_action: np.ndarray
    deep_identity: DeepNodeIdentity


def pi05_p2_identity_metrics(
    *,
    extractor: torch.Tensor,
    embed_image: torch.Tensor,
    prefix: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Validate the three authoritative PI0.5 P2 paths without rescaling."""

    values = {
        "phase2_extractor": extractor,
        "direct_embed_image": embed_image,
        "official_prefix_base_slice": prefix,
    }
    for name, value in values.items():
        _validate_feature(value, name=name, width=PI05_P2_WIDTH)
    reference = values["direct_embed_image"].float()
    reference_norm = torch.linalg.vector_norm(reference)
    if not bool(torch.isfinite(reference_norm)) or float(reference_norm) == 0.0:
        raise ActionRepresentationError("direct embed_image P2 norm must be positive")
    comparisons: dict[str, Any] = {}
    for name in ("phase2_extractor", "official_prefix_base_slice"):
        candidate = values[name].float()
        difference = candidate - reference
        metrics = {
            "max_abs_difference": float(difference.abs().max()),
            "mean_abs_difference": float(difference.abs().mean()),
            "relative_l2_error": float(
                torch.linalg.vector_norm(difference) / reference_norm
            ),
        }
        metrics["within_tolerance"] = bool(
            torch.allclose(candidate, reference, atol=atol, rtol=rtol)
        )
        comparisons[name] = metrics
    result = {
        "definition_id": PI05_P2_DEFINITION_ID,
        "shape": list(extractor.shape),
        "dtype": {name: str(value.dtype) for name, value in values.items()},
        "l2_norm": {
            name: float(torch.linalg.vector_norm(value.float()))
            for name, value in values.items()
        },
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparisons_to_embed_image": comparisons,
        "identity_pass": all(
            value["within_tolerance"] for value in comparisons.values()
        ),
        "manual_scaling": "none",
    }
    if not result["identity_pass"]:
        raise ActionRepresentationError(f"PI0.5 P2 identity mismatch: {result}")
    return result


def midpoint_layer_index(total_layers: int) -> int:
    """Select the last decoder block in the first half of a tower."""

    if type(total_layers) is not int or total_layers < 2:
        raise ActionRepresentationError(
            "decoder tower must contain at least two layers"
        )
    return total_layers // 2 - 1


def _tensor_output(output: Any, *, name: str) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise ActionRepresentationError(f"{name} hook did not capture a tensor")
    return value


def _validate_feature(value: torch.Tensor, *, name: str, width: int) -> torch.Tensor:
    if tuple(value.shape) != (1, NUM_VISUAL_TOKENS, width):
        raise ActionRepresentationError(
            f"{name} must have shape [1,{NUM_VISUAL_TOKENS},{width}], "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ActionRepresentationError(f"{name} contains non-finite values")
    return value


def _validate_action(action: Any, *, name: str) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
        raise ActionRepresentationError(f"{name} must be finite shape [7]")
    return value


def extract_openvla_action_representation(
    *,
    model: Any,
    model_inputs: dict[str, torch.Tensor],
    unnorm_key: str,
    deploy_action: Callable[[np.ndarray], np.ndarray],
) -> ModelActionRepresentation:
    """Decode one action while capturing O2 and a midpoint visual-token state."""

    try:
        layers = model.language_model.model.layers
        projector = model.projector
    except AttributeError as error:
        raise ActionRepresentationError(
            "OpenVLA does not expose projector and Llama decoder layers"
        ) from error
    total_layers = len(layers)
    layer_index = midpoint_layer_index(total_layers)
    projected: list[torch.Tensor] = []
    deep: list[torch.Tensor] = []

    def capture_projected(_module: Any, _inputs: Any, output: Any) -> None:
        projected.append(_tensor_output(output, name="OpenVLA O2"))

    def capture_deep(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = _tensor_output(output, name="OpenVLA O-deep")
        # Generation revisits the layer with one cached token. Only the initial
        # multimodal call contains the 256 visual-token span after BOS.
        if hidden.ndim == 3 and hidden.shape[1] >= NUM_VISUAL_TOKENS + 1:
            deep.append(hidden[:, 1 : NUM_VISUAL_TOKENS + 1, :])

    handles = (
        projector.register_forward_hook(capture_projected),
        layers[layer_index].register_forward_hook(capture_deep),
    )
    try:
        with torch.inference_mode():
            raw_action = model.predict_action(
                **model_inputs, unnorm_key=unnorm_key, do_sample=False
            )
    finally:
        for handle in reversed(handles):
            handle.remove()

    if len(projected) != 1:
        raise ActionRepresentationError(
            f"OpenVLA O2 capture count must be 1, got {len(projected)}"
        )
    if len(deep) != 1:
        raise ActionRepresentationError(
            f"OpenVLA O-deep full-sequence capture count must be 1, got {len(deep)}"
        )
    o2 = _validate_feature(projected[0], name="OpenVLA O2", width=4096)
    o_deep = _validate_feature(deep[0], name="OpenVLA O-deep", width=4096)
    deployed = _validate_action(
        deploy_action(np.asarray(raw_action).copy()), name="OpenVLA deployed action"
    )
    return ModelActionRepresentation(
        projected=o2,
        deep=o_deep,
        deployed_action=deployed,
        deep_identity=DeepNodeIdentity(
            module_path=f"language_model.model.layers[{layer_index}]",
            total_layers=total_layers,
            zero_based_layer_index=layer_index,
            one_based_layer_index=layer_index + 1,
            capture_point="decoder block output",
            visual_token_slice="multimodal hidden_state[:, 1:257, :]",
        ),
    )


def extract_pi05_action_representation(
    *,
    policy: Any,
    model: Any,
    raw_observation: dict[str, Any],
    noise: np.ndarray,
    authoritative_p2_provider: Callable[[], torch.Tensor],
) -> ModelActionRepresentation:
    """Infer one PI0.5 action chunk and pair it with official base-camera P2."""

    try:
        paligemma = model.paligemma_with_expert.paligemma
        layers = paligemma.language_model.layers
    except AttributeError as error:
        raise ActionRepresentationError(
            "PI0Pytorch does not expose PaliGemma projector and prefix layers"
        ) from error
    total_layers = len(layers)
    layer_index = midpoint_layer_index(total_layers)
    prefix_hidden: list[torch.Tensor] = []

    def capture_deep(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = _tensor_output(output, name="PI0.5 P-deep")
        if hidden.ndim == 3 and hidden.shape[1] >= 3 * NUM_VISUAL_TOKENS:
            prefix_hidden.append(hidden[:, :NUM_VISUAL_TOKENS, :])

    handles = (layers[layer_index].register_forward_hook(capture_deep),)
    try:
        with torch.inference_mode():
            output = policy.infer(raw_observation, noise=noise)
    finally:
        for handle in reversed(handles):
            handle.remove()

    if len(prefix_hidden) != 1:
        raise ActionRepresentationError(
            f"PI0.5 P-deep prefix capture count must be 1, got {len(prefix_hidden)}"
        )
    hidden_size = int(paligemma.config.text_config.hidden_size)
    if hidden_size != PI05_P2_WIDTH:
        raise ActionRepresentationError(
            f"PI0.5 PaliGemma hidden size must be 2048, got {hidden_size}"
        )
    if not callable(authoritative_p2_provider):
        raise ActionRepresentationError("PI0.5 P2 provider must be callable")
    with torch.inference_mode():
        p2 = _validate_feature(
            authoritative_p2_provider().clone(), name="PI0.5 P2", width=2048
        )
    # PI0Pytorch inference is torch.compile'd with CUDA Graphs. Decoder-layer
    # outputs can therefore alias a static output buffer that the next infer()
    # call overwrites. Take ownership after infer() returns (outside the
    # compiled region) so repeated extraction and serialization remain valid.
    p_deep = _validate_feature(
        prefix_hidden[0].clone(), name="PI0.5 P-deep", width=2048
    )
    actions = output["actions"] if isinstance(output, dict) else output
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not len(actions):
        raise ActionRepresentationError("PI0.5 deployed action chunk must be [H,7]")
    deployed = _validate_action(actions[0], name="PI0.5 first deployed action")
    return ModelActionRepresentation(
        projected=p2,
        deep=p_deep,
        deployed_action=deployed,
        deep_identity=DeepNodeIdentity(
            module_path=(
                f"paligemma_with_expert.paligemma.language_model.layers[{layer_index}]"
            ),
            total_layers=total_layers,
            zero_based_layer_index=layer_index,
            one_based_layer_index=layer_index + 1,
            capture_point="PaliGemma prefix decoder block output",
            visual_token_slice="prefix hidden_state[:, 0:256, :] (base_0_rgb)",
        ),
    )


def extract_pi05_official_p2(*, model: Any, observation: Any) -> torch.Tensor:
    """Extract current PI0Pytorch base-camera ``embed_image()`` P2."""

    with torch.inference_mode():
        prepared = model._preprocess_observation(observation, train=False)
        if not isinstance(prepared, tuple) or len(prepared) != 5:
            raise ActionRepresentationError(
                "PI0Pytorch preprocessing must return the five-value contract"
            )
        images = prepared[0]
        if not isinstance(images, list) or len(images) != 3:
            raise ActionRepresentationError(
                "PI0Pytorch preprocessing must preserve three image slots"
            )
        value = model.paligemma_with_expert.embed_image(images[0]).clone()
    return _validate_feature(value, name="PI0.5 official P2", width=PI05_P2_WIDTH)
