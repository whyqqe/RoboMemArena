"""Independent Verifier — VLA stop ≠ action success.

Outcomes: VERIFIED | FAILED | AMBIGUOUS.
Phase-1 uses harness-reported stage completions as the sound verification signal
(SEEN ≠ VERIFIED). Phase-3 adds vision / drop / collision classifiers behind
MEMEXP_BOLT_VISION_VERIFY.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .types import (
    CS_COMMITTED,
    CS_OBSERVED,
    VR_AMBIGUOUS,
    VR_FAILED,
    VR_VERIFIED,
    ActionContract,
    VerifyResult,
)

if TYPE_CHECKING:
    from .harness import BoltHarness

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def vision_verify_on() -> bool:
    return str(os.environ.get("MEMEXP_BOLT_VISION_VERIFY", "0")).strip().lower() in _TRUTHY


class IndependentVerifier:
    def __init__(self) -> None:
        self.n_verified = 0
        self.n_failed = 0
        self.n_ambiguous = 0

    def verify(
        self,
        harness: "BoltHarness",
        contract: ActionContract,
        *,
        verified_stages: list[str] | None = None,
        action_trace: dict[str, Any] | None = None,
    ) -> VerifyResult:
        trace = action_trace or {}
        stages = set(verified_stages or [])

        # explicit physical failure tags from the harness / actuator guard
        phys = str(trace.get("physical_failure") or "").strip()
        if phys:
            self.n_failed += 1
            return VerifyResult(
                outcome=VR_FAILED,
                reason=f"physical:{phys}",
                node_id=contract.node_id,
                physical_failure=phys,
            )

        node = harness.graph.nodes.get(contract.node_id) if contract.node_id else None
        stage = (node.verifies_stage if node else "") or ""

        if stage and stage in stages:
            self.n_verified += 1
            updates = []
            if node and node.success_observation:
                updates.append({
                    "predicate": node.success_observation,
                    "status": CS_COMMITTED,
                    "confidence": 0.95,
                    "source": [f"stage:{stage}"],
                })
            return VerifyResult(
                outcome=VR_VERIFIED,
                reason=f"stage verified: {stage}",
                node_id=contract.node_id,
                claim_updates=updates,
            )

        # grasp may settle derivationally when its settles_with stage is verified
        if node and node.settles_with and node.settles_with in stages:
            self.n_verified += 1
            updates = []
            if node.success_observation:
                updates.append({
                    "predicate": node.success_observation,
                    "status": CS_COMMITTED,
                    "confidence": 0.9,
                    "source": [f"derived:{node.settles_with}"],
                })
            return VerifyResult(
                outcome=VR_VERIFIED,
                reason=f"derivational settle via {node.settles_with}",
                node_id=contract.node_id,
                claim_updates=updates,
            )

        # Phase-3 vision stub: if enabled and trace carries a vision_score
        if vision_verify_on() and "vision_score" in trace:
            score = float(trace.get("vision_score") or 0.0)
            if score >= 0.8:
                self.n_verified += 1
                return VerifyResult(
                    outcome=VR_VERIFIED,
                    reason=f"vision_score={score:.2f}",
                    node_id=contract.node_id,
                    claim_updates=[{
                        "predicate": contract.success_observation or contract.predicted_transition,
                        "status": CS_OBSERVED,
                        "confidence": score,
                        "source": ["vision"],
                    }],
                )
            if score <= 0.2:
                self.n_failed += 1
                return VerifyResult(
                    outcome=VR_FAILED,
                    reason=f"vision_score={score:.2f}",
                    node_id=contract.node_id,
                    physical_failure="vision_reject",
                )

        # no positive evidence → AMBIGUOUS (do NOT settle)
        self.n_ambiguous += 1
        return VerifyResult(
            outcome=VR_AMBIGUOUS,
            reason="no independent verification signal",
            node_id=contract.node_id,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_verified": self.n_verified,
            "n_failed": self.n_failed,
            "n_ambiguous": self.n_ambiguous,
        }
