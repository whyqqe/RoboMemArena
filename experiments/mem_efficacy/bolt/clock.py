"""Dual-clock synchronization (ReSync-style commitment–evidence gap).

cognition_age  = now - last_planner_update
action_age     = now - action_contract_creation
evidence_age   = now - newest_supporting_observation
commitment_gap = action_age - evidence_age

When gap ∈ [g_min, g_max]: continue the current action chunk.
When gap > g_max: HOLD new commits and refresh minimal evidence.
When gap < g_min but the world is still uncertain: prefer OBSERVE over ACT.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


def _fenv(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


@dataclass
class SyncState:
    last_planner_update: float = 0.0
    last_contract_creation: float = 0.0
    newest_evidence: float = 0.0
    last_tick: float = 0.0
    # supported window for commitment_gap
    g_min: float = field(default_factory=lambda: _fenv("MEMEXP_BOLT_GAP_MIN", 0.0))
    g_max: float = field(default_factory=lambda: _fenv("MEMEXP_BOLT_GAP_MAX", 30.0))
    # counters
    n_in_window: int = 0
    n_gap_high: int = 0
    n_gap_low: int = 0
    n_hold: int = 0
    n_observe_preferred: int = 0

    def mark_planner(self, now: float | None = None) -> None:
        self.last_planner_update = float(now if now is not None else time.time())

    def mark_contract(self, now: float | None = None) -> None:
        self.last_contract_creation = float(now if now is not None else time.time())

    def mark_evidence(self, now: float | None = None) -> None:
        self.newest_evidence = float(now if now is not None else time.time())

    def ages(self, now: float | None = None) -> dict[str, float]:
        now = float(now if now is not None else time.time())
        self.last_tick = now
        cog = now - self.last_planner_update if self.last_planner_update else float("inf")
        act = now - self.last_contract_creation if self.last_contract_creation else float("inf")
        evi = now - self.newest_evidence if self.newest_evidence else float("inf")
        # commitment_gap = action_age - evidence_age; undefined when either is missing
        if self.last_contract_creation and self.newest_evidence:
            gap = act - evi
        elif self.last_contract_creation:
            gap = act  # evidence never arrived → treat as large gap
        else:
            gap = 0.0
        return {
            "now": now,
            "cognition_age": cog,
            "action_age": act,
            "evidence_age": evi,
            "commitment_gap": gap,
            "g_min": self.g_min,
            "g_max": self.g_max,
        }

    def window_status(self, now: float | None = None, *, world_uncertain: bool = False) -> str:
        """Return 'ok' | 'gap_high' | 'gap_low_observe'."""
        a = self.ages(now)
        gap = a["commitment_gap"]
        if gap > self.g_max:
            self.n_gap_high += 1
            return "gap_high"
        if gap < self.g_min and world_uncertain:
            self.n_gap_low += 1
            self.n_observe_preferred += 1
            return "gap_low_observe"
        self.n_in_window += 1
        return "ok"

    def allow_commit(self, now: float | None = None, *, world_uncertain: bool = False) -> tuple[bool, str]:
        status = self.window_status(now, world_uncertain=world_uncertain)
        if status == "gap_high":
            self.n_hold += 1
            return False, "commitment_gap above supported window — refresh evidence"
        if status == "gap_low_observe":
            return False, "commitment_gap below window with uncertain world — prefer observe"
        return True, "in window"

    def snapshot(self) -> dict[str, Any]:
        a = self.ages()
        return {
            **{k: (None if v == float("inf") else round(v, 3)) for k, v in a.items()},
            "n_in_window": self.n_in_window,
            "n_gap_high": self.n_gap_high,
            "n_gap_low": self.n_gap_low,
            "n_hold": self.n_hold,
            "n_observe_preferred": self.n_observe_preferred,
        }
