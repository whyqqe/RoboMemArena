"""Verified Ledger — monotone progress that only SETTLES on verified success.

Counts (`count:place`, `count:pour`, `count:verified_success`) are ALWAYS
derived from the ledger, never from VLM self-report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    step: int
    node_id: str
    primitive: str
    outcome: str = ""          # VERIFIED | FAILED | AMBIGUOUS | ""
    reason: str = ""
    physical_failure: str = ""


class VerifiedLedger:
    """`L_t` — attempt / success / failure log + derived counters."""

    def __init__(self) -> None:
        self.attempts: list[AttemptRecord] = []
        self.verified_stages: list[str] = []
        self.n_verified = 0
        self.n_failed = 0
        self.n_ambiguous = 0
        self.n_attempts = 0
        self.by_phase_verified: dict[str, int] = {}
        self.by_phase_attempted: dict[str, int] = {}
        self.failure_reasons: dict[str, int] = {}

    def record_attempt(self, *, step: int, node_id: str, primitive: str, phase: str = "") -> AttemptRecord:
        rec = AttemptRecord(step=step, node_id=node_id, primitive=primitive)
        self.attempts.append(rec)
        self.n_attempts += 1
        if phase:
            self.by_phase_attempted[phase] = self.by_phase_attempted.get(phase, 0) + 1
        return rec

    def record_outcome(
        self,
        *,
        node_id: str,
        outcome: str,
        reason: str = "",
        physical_failure: str = "",
        phase: str = "",
        stage_name: str = "",
    ) -> None:
        # update last matching attempt
        for rec in reversed(self.attempts):
            if rec.node_id == node_id and not rec.outcome:
                rec.outcome = outcome
                rec.reason = reason
                rec.physical_failure = physical_failure
                break
        if outcome == "VERIFIED":
            self.n_verified += 1
            if phase:
                self.by_phase_verified[phase] = self.by_phase_verified.get(phase, 0) + 1
            if stage_name and stage_name not in self.verified_stages:
                self.verified_stages.append(stage_name)
        elif outcome == "FAILED":
            self.n_failed += 1
            key = physical_failure or reason or "unknown"
            self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
        elif outcome == "AMBIGUOUS":
            self.n_ambiguous += 1

    # -- derived counts (NEVER from VLM self-report) ---------------------------------------
    def count(self, kind: str) -> int:
        k = str(kind or "").lower().replace("count:", "")
        if k in {"verified_success", "verified", "success"}:
            return int(self.n_verified)
        if k in {"place", "places"}:
            return int(self.by_phase_verified.get("place", 0))
        if k in {"pour", "pours"}:
            return int(self.by_phase_verified.get("pour", 0)
                       + self.by_phase_verified.get("pour_one", 0)
                       + self.by_phase_verified.get("pour_two", 0))
        if k in {"grasp", "pick", "picks"}:
            return int(self.by_phase_verified.get("grasp", 0))
        if k in {"attempt", "attempts"}:
            return int(self.n_attempts)
        if k in {"fail", "failed", "failure"}:
            return int(self.n_failed)
        return int(self.by_phase_verified.get(k, 0))

    def derived_board(self) -> list[str]:
        """Progress rows.  Labelled so the tokens are not executable commands.

        `count:grasp = 0` is a two-word line whose first token is an imperative verb and
        whose second is a phase name; paired with the object elsewhere on the board it
        invites the same copy-the-identifier failure the OPEN list produced.  The labels
        below are deliberately non-imperative.
        """
        rows = []
        for kind, label in (
            ("grasp", "grasp stages verified"),
            ("place", "place stages verified"),
            ("pour", "pour stages verified"),
            ("verified_success", "scored stages verified"),
        ):
            rows.append(f"  {label} = {self.count(kind)}")
        return rows

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_attempts": self.n_attempts,
            "n_verified": self.n_verified,
            "n_failed": self.n_failed,
            "n_ambiguous": self.n_ambiguous,
            "by_phase_verified": dict(self.by_phase_verified),
            "by_phase_attempted": dict(self.by_phase_attempted),
            "failure_reasons": dict(self.failure_reasons),
            "verified_stages": list(self.verified_stages),
            "derived": {f"count:{k}": self.count(k)
                        for k in ("grasp", "place", "pour", "verified_success", "attempts", "fail")},
        }
