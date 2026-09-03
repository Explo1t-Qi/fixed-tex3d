"""Regression tests for OpenVLA generation input alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROBOT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2] / "openvla/experiments/robot"
)
sys.path.insert(0, str(ROBOT_EXPERIMENT_DIR))

from openvla_model_inputs import (  # noqa: E402
    LLAMA_EMPTY_TOKEN_ID,
    ensure_trailing_empty_token,
)


def test_missing_empty_token_extends_ids_and_attention_mask_together() -> None:
    pixel_values = torch.zeros((1, 6, 2, 2), dtype=torch.bfloat16)
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": pixel_values,
    }

    returned = ensure_trailing_empty_token(inputs)

    assert returned is inputs
    assert inputs["input_ids"].tolist() == [[1, 2, LLAMA_EMPTY_TOKEN_ID]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    assert inputs["pixel_values"] is pixel_values


def test_existing_empty_token_is_not_duplicated() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, LLAMA_EMPTY_TOKEN_ID]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }

    ensure_trailing_empty_token(inputs)

    assert inputs["input_ids"].tolist() == [[1, LLAMA_EMPTY_TOKEN_ID]]
    assert inputs["attention_mask"].tolist() == [[1, 1]]
