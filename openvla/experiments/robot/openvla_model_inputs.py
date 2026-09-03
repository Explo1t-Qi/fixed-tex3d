"""Small, dependency-light helpers for OpenVLA generation inputs."""

from __future__ import annotations

from typing import Final, MutableMapping, TypeVar

import torch


LLAMA_EMPTY_TOKEN_ID: Final[int] = 29_871
ModelInputsT = TypeVar(
    "ModelInputsT", bound=MutableMapping[str, torch.Tensor]
)


def ensure_trailing_empty_token(inputs: ModelInputsT) -> ModelInputsT:
    """Append OpenVLA's LLaMA empty token and its attention entry together.

    The checkpoint model's ``predict_action`` appends token 29871 when it is
    absent, but it does not extend a caller-provided attention mask.  Doing the
    equivalent operation at the call boundary prevents a one-token shape
    mismatch while leaving every non-text input untouched.
    """
    if "input_ids" not in inputs:
        return inputs

    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty rank-2 tensor")
    if bool(torch.all(input_ids[:, -1] == LLAMA_EMPTY_TOKEN_ID).item()):
        return inputs

    batch_size = input_ids.shape[0]
    trailing_token = torch.full(
        (batch_size, 1),
        LLAMA_EMPTY_TOKEN_ID,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    inputs["input_ids"] = torch.cat((input_ids, trailing_token), dim=1)

    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"]
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids before token alignment"
            )
        trailing_attention = torch.ones(
            (batch_size, 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs["attention_mask"] = torch.cat(
            (attention_mask, trailing_attention), dim=1
        )

    return inputs
