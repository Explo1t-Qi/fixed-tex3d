#!/usr/bin/env python3
"""Run Phase 3 action-predictive O2/P2 texture optimization."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2_shared_optimization import (  # noqa: E402
    ACTION_PREDICTIVE_GRADIENT_ENSEMBLE_OBJECTIVE,
    main as optimization_main,
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--objective" in arguments:
        raise ValueError("Phase 3 wrapper fixes the objective; omit --objective")
    return optimization_main(
        ["--objective", ACTION_PREDICTIVE_GRADIENT_ENSEMBLE_OBJECTIVE, *arguments]
    )


if __name__ == "__main__":
    raise SystemExit(main())
