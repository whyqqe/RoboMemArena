"""Claim schema and BeliefState.

A Claim is a structured, invalidate-able fact — the opposite of an untraceable
prompt summary. High-risk actions may only use OBSERVED / COMMITTED claims;
BELIEVED supports low-risk only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .types import (
    CS_BELIEVED,
    CS_COMMITTED,
    CS_CONTRADICTED,
    CS_OBSERVED,
    CS_STALE,
    RISK_HIGH,
    RISK_LOW,
    RISK_MID,
)


@dataclass
class Claim:
    claim_id: str
    predicate: str
    value: Any = True
    status: str = CS_BELIEVED
    confidence: float = 0.5
    source: list[str] = field(default_factory=list)
    valid_from: float = 0.0
    invalidation: list[str] = field(default_factory=list)
    required_for: list[str] = field(default_factory=list)
    last_checked: float = 0.0
    n_strengthen: int = 0
    n_invalidate: int = 0

    def supports(self, risk: str) -> bool:
        """Whether this claim is strong enough for the given risk level."""
        if self.status in {CS_STALE, CS_CONTRADICTED}:
            return False
        if risk == RISK_HIGH:
            return self.status in {CS_OBSERVED, CS_COMMITTED} and self.confidence >= 0.8
        if risk == RISK_MID:
            return self.status in {CS_OBSERVED, CS_COMMITTED, CS_BELIEVED} and self.confidence >= 0.5
        return self.status != CS_CONTRADICTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "predicate": self.predicate,
            "value": self.value,
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "source": list(self.source),
            "valid_from": self.valid_from,
            "invalidation": list(self.invalidation),
            "required_for": list(self.required_for),
            "n_strengthen": self.n_strengthen,
            "n_invalidate": self.n_invalidate,
        }


class BeliefState:
    """The `b_t` component of S_t."""

    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self._next = 1
        self.events: list[dict[str, Any]] = []  # add / strengthen / invalidate / contradict
        self.n_unsupported_use = 0
        self.n_stale_use = 0

    def _mint(self) -> str:
        cid = f"c_{self._next}"
        self._next += 1
        return cid

    def add(
        self,
        predicate: str,
        *,
        value: Any = True,
        status: str = CS_BELIEVED,
        confidence: float = 0.5,
        source: list[str] | None = None,
        invalidation: list[str] | None = None,
        required_for: list[str] | None = None,
        now: float | None = None,
    ) -> Claim:
        now = float(now if now is not None else time.time())
        existing = self.find(predicate)
        if existing is not None:
            return self.strengthen(existing.claim_id, confidence=confidence, source=source, status=status, now=now)
        c = Claim(
            claim_id=self._mint(),
            predicate=predicate,
            value=value,
            status=status,
            confidence=float(confidence),
            source=list(source or []),
            valid_from=now,
            invalidation=list(invalidation or []),
            required_for=list(required_for or []),
            last_checked=now,
        )
        self.claims[c.claim_id] = c
        self.events.append({"op": "add", "claim_id": c.claim_id, "predicate": predicate, "t": now})
        return c

    def find(self, predicate: str) -> Claim | None:
        for c in self.claims.values():
            if c.predicate == predicate and c.status not in {CS_CONTRADICTED}:
                return c
        return None

    def strengthen(
        self,
        claim_id: str,
        *,
        confidence: float | None = None,
        source: list[str] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> Claim:
        c = self.claims[claim_id]
        now = float(now if now is not None else time.time())
        if confidence is not None:
            c.confidence = max(float(c.confidence), float(confidence))
        if source:
            for s in source:
                if s not in c.source:
                    c.source.append(s)
        if status:
            # never demote COMMITTED → BELIEVED via strengthen
            rank = {CS_BELIEVED: 1, CS_OBSERVED: 2, CS_COMMITTED: 3, CS_STALE: 0, CS_CONTRADICTED: -1}
            if rank.get(status, 0) >= rank.get(c.status, 0):
                c.status = status
        c.n_strengthen += 1
        c.last_checked = now
        self.events.append({"op": "strengthen", "claim_id": claim_id, "t": now})
        return c

    def invalidate(self, claim_id: str, *, reason: str = "", now: float | None = None) -> Claim:
        c = self.claims[claim_id]
        now = float(now if now is not None else time.time())
        c.status = CS_STALE
        c.n_invalidate += 1
        c.last_checked = now
        self.events.append({"op": "invalidate", "claim_id": claim_id, "reason": reason, "t": now})
        return c

    def contradict(self, claim_id: str, *, reason: str = "", now: float | None = None) -> Claim:
        c = self.claims[claim_id]
        now = float(now if now is not None else time.time())
        c.status = CS_CONTRADICTED
        c.confidence = 0.0
        c.last_checked = now
        self.events.append({"op": "contradict", "claim_id": claim_id, "reason": reason, "t": now})
        return c

    def check_invalidation_triggers(self, triggers: list[str], *, now: float | None = None) -> list[str]:
        """Mark any claim whose invalidation list intersects `triggers` as STALE."""
        hit: list[str] = []
        trig = set(triggers or [])
        if not trig:
            return hit
        for c in list(self.claims.values()):
            if c.status in {CS_STALE, CS_CONTRADICTED}:
                continue
            if any(t in trig for t in c.invalidation):
                self.invalidate(c.claim_id, reason=f"trigger:{sorted(trig & set(c.invalidation))}", now=now)
                hit.append(c.claim_id)
        return hit

    def precondition_ok(self, predicates: list[str], *, risk: str = RISK_MID) -> tuple[bool, str]:
        """Return (ok, reason). Counts unsupported / stale uses for telemetry."""
        for pred in predicates or []:
            c = self.find(pred)
            if c is None:
                self.n_unsupported_use += 1
                return False, f"missing claim: {pred}"
            if c.status == CS_STALE:
                self.n_stale_use += 1
                return False, f"stale claim: {pred}"
            if not c.supports(risk):
                self.n_unsupported_use += 1
                return False, f"claim {pred} status={c.status} conf={c.confidence:.2f} insufficient for risk={risk}"
        return True, ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_claims": len(self.claims),
            "by_status": {
                s: sum(1 for c in self.claims.values() if c.status == s)
                for s in (CS_OBSERVED, CS_BELIEVED, CS_COMMITTED, CS_STALE, CS_CONTRADICTED)
            },
            "n_unsupported_use": self.n_unsupported_use,
            "n_stale_use": self.n_stale_use,
            "n_events": len(self.events),
            "claims": [c.to_dict() for c in self.claims.values()],
        }
