"""Rule-based Evidence Router (Phase-2 learned router is a closable stub).

Chooses among: read Claim / recall history / observe physically / act.
Utility:
  U(m) = E[ΔΦ | m] - λ_r Risk(m) - λ_l Latency(m) - λ_c ContextCost(m)

Safety floor: if the evidence channel has been starved and the active
obligation has not advanced, emit one minimal evidence request.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .types import AV_ACT, AV_OBSERVE, AV_RECALL, DecisionNeed

if TYPE_CHECKING:
    from .harness import BoltHarness

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def router_on() -> bool:
    return str(os.environ.get("MEMEXP_BOLT_ROUTER", "1")).strip().lower() in _TRUTHY


def floor_on() -> bool:
    return str(os.environ.get("MEMEXP_BOLT_FLOOR", "1")).strip().lower() in _TRUTHY


class EvidenceRouter:
    def __init__(self) -> None:
        self.n_recall = 0
        self.n_observe = 0
        self.n_act = 0
        self.n_floor_fire = 0
        self.n_starved_steps = 0
        self._steps_since_evidence = 0
        self.lambda_r = float(os.environ.get("MEMEXP_BOLT_LAMBDA_R", "0.3") or 0.3)
        self.lambda_l = float(os.environ.get("MEMEXP_BOLT_LAMBDA_L", "0.1") or 0.1)
        self.lambda_c = float(os.environ.get("MEMEXP_BOLT_LAMBDA_C", "0.05") or 0.05)
        self.starvation_limit = int(os.environ.get("MEMEXP_BOLT_STARVE_LIMIT", "4") or 4)

    def route(self, harness: "BoltHarness", need: DecisionNeed) -> dict[str, Any]:
        """Return {mode, reason, u_scores}."""
        if not router_on():
            self.n_act += 1
            return {"mode": AV_ACT, "reason": "router off", "u_scores": {}}

        # safety floor: evidence starvation
        if floor_on() and self._steps_since_evidence >= self.starvation_limit:
            active = harness.graph.active()
            if active is not None and active.is_open():
                self.n_floor_fire += 1
                self.n_observe += 1
                self._steps_since_evidence = 0
                return {
                    "mode": AV_OBSERVE,
                    "reason": f"evidence starvation floor (>{self.starvation_limit} steps)",
                    "u_scores": {},
                }

        scores = {
            AV_ACT: self._u_act(harness, need),
            AV_RECALL: self._u_recall(harness, need),
            AV_OBSERVE: self._u_observe(harness, need),
        }
        best = max(scores, key=scores.get)
        if best == AV_RECALL:
            self.n_recall += 1
        elif best == AV_OBSERVE:
            self.n_observe += 1
            self._steps_since_evidence = 0
        else:
            self.n_act += 1
            self._steps_since_evidence += 1
            self.n_starved_steps = self._steps_since_evidence
        return {"mode": best, "reason": f"argmax U; need={need.reason}", "u_scores": scores}

    def note_evidence_served(self) -> None:
        self._steps_since_evidence = 0

    def _u_act(self, harness: "BoltHarness", need: DecisionNeed) -> float:
        # Expected ΔΦ ≈ weight of active node if deps ready, else near-zero
        active = harness.graph.active()
        if active is None:
            return -1.0
        if not harness.graph.deps_ready(active):
            return -0.5
        delta = float(active.weight)
        risk = 0.4 if need.risk == "high" else 0.1
        return delta - self.lambda_r * risk - self.lambda_l * 0.2

    def _u_recall(self, harness: "BoltHarness", need: DecisionNeed) -> float:
        if not need.unknowns:
            return -1.0
        # recalling helps when unknowns are historical (dep already attempted elsewhere)
        hist = sum(1 for u in need.unknowns if u.startswith("failure_mode:") or u.startswith("dep:"))
        if hist <= 0:
            return -0.2
        return 0.6 * hist - self.lambda_c * 0.5 - self.lambda_l * 0.3

    def _u_observe(self, harness: "BoltHarness", need: DecisionNeed) -> float:
        if any(u.startswith("look:") or u.startswith("sync:") for u in need.unknowns):
            return 0.9 - self.lambda_l * 0.5
        if need.minimum_evidence in {"current_view", "keyframe_or_current_view"} and need.unknowns:
            return 0.5 - self.lambda_l * 0.4
        return -0.3

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_recall": self.n_recall,
            "n_observe": self.n_observe,
            "n_act": self.n_act,
            "n_floor_fire": self.n_floor_fire,
            "n_starved_steps": self.n_starved_steps,
        }
