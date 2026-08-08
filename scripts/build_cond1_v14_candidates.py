#!/usr/bin/env python3
"""Generate the six-family V14 controller candidate set from frozen V13 JSON."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_v14_pilot import CORE6, enable_v14_state  # noqa: E402


SOURCE = (
    ROOT
    / "src/Bayesian_state/problems/modules/hypo_transition/candidates"
    / "hypo_transition_profile_v13_candidates.json"
)
TARGET = (
    ROOT
    / "src/Bayesian_state/problems/modules/hypo_transition/candidates"
    / "hypo_transition_profile_v14_candidates.json"
)
GAINS = (0.20, 0.35, 0.50)


def short_id(candidate_id: str) -> str:
    prefix = "c1_v13_"
    if not candidate_id.startswith(prefix):
        raise ValueError(candidate_id)
    return candidate_id[len(prefix) :]


def main() -> None:
    candidates = json.loads(SOURCE.read_text(encoding="utf-8"))["cond1_v13"]
    selected = [item for item in candidates if item["id"] in CORE6]
    if {item["id"] for item in selected} != CORE6:
        raise RuntimeError("V13 source does not contain the expected core6 controllers")

    output = []
    for item in selected:
        family = short_id(str(item["id"]))
        output.append(
            {
                "id": f"c1_v14_{family}_state_off",
                "family_id": f"c1_v14_{family}",
                "state_mode": "off",
                "description": (
                    f"V14 core family {family}; exact V13 controller behavior, "
                    "retained as the no-persistence ablation."
                ),
                "hypo_transitions_kwargs": deepcopy(item["hypo_transitions_kwargs"]),
            }
        )
        for gain in GAINS:
            gain_id = f"{gain:.2f}".replace(".", "p")
            output.append(
                {
                    "id": f"c1_v14_{family}_state_g{gain_id}",
                    "family_id": f"c1_v14_{family}",
                    "state_mode": "confidence_weighted_error",
                    "state_error_gain": gain,
                    "description": (
                        f"V14 core family {family}; persistent confidence-weighted "
                        f"belief instability with error gain {gain:.2f}."
                    ),
                    "hypo_transitions_kwargs": enable_v14_state(
                        item["hypo_transitions_kwargs"],
                        error_gain=gain,
                        decay=0.80,
                        threshold=0.55,
                    ),
                }
            )

    payload = {
        "schema_version": "hypo_transition_profile_candidates.v1",
        "source": SOURCE.name,
        "design": {
            "controller_families": 6,
            "state_settings_per_family": 4,
            "inner_strategy_states": [
                "conservative",
                "stable",
                "aggressive",
                "stubborn",
            ],
            "note": (
                "The legacy 16-controller V13 file is preserved. V14 searches six "
                "supported families and treats state gain as a within-family setting."
            ),
        },
        "cond1_v14": output,
    }
    TARGET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output)} configurations across 6 families -> {TARGET}")


if __name__ == "__main__":
    main()
