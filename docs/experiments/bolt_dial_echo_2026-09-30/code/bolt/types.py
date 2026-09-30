"""Shared types for BOLT-Sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- Claim status -------------------------------------------------------------------------
CS_OBSERVED = "OBSERVED"
CS_BELIEVED = "BELIEVED"
CS_COMMITTED = "COMMITTED"
CS_CONTRADICTED = "CONTRADICTED"
CS_STALE = "STALE"

# --- Obligation node status ---------------------------------------------------------------
NS_OPEN = "OPEN"
NS_ATTEMPTED = "ATTEMPTED"
NS_AMB = "AMBIGUOUS"
NS_FAILED = "FAILED"
NS_SETTLED = "SETTLED"

# --- Dependency hardness ------------------------------------------------------------------
DEP_SOFT = "soft"   # grasp→place: attempted is enough
DEP_HARD = "hard"   # pour_1→pour_2 / place_i→place_{i+1}: verified SETTLED required

# --- Risk ---------------------------------------------------------------------------------
RISK_LOW = "low"
RISK_MID = "mid"
RISK_HIGH = "high"

# --- Verify outcomes ----------------------------------------------------------------------
VR_VERIFIED = "VERIFIED"
VR_FAILED = "FAILED"
VR_AMBIGUOUS = "AMBIGUOUS"

# --- Arbiter verdicts ---------------------------------------------------------------------
AV_ACT = "act"
AV_RECALL = "recall"
AV_OBSERVE = "observe"
AV_RETRY = "retry"
AV_RECOVER = "recover"
AV_HOLD = "hold"          # dual-clock: gap outside window
AV_REWRITE = "rewrite"    # rejected → corrected to ACTIVE obligation


@dataclass
class ActionContract:
    """Structured candidate emitted by the planner (or rewritten by the Arbiter)."""
    primitive: str = ""
    object: str = ""
    target: str = ""
    preconditions: list[str] = field(default_factory=list)
    predicted_transition: str = ""
    success_observation: str = ""
    abort_conditions: list[str] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    risk_level: str = RISK_MID
    node_id: str = ""             # bound obligation node, if known
    geometry: list[str] = field(default_factory=list)  # Phase-3 ReKep stubs
    raw_text: str = ""            # original planner output
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "object": self.object,
            "target": self.target,
            "preconditions": list(self.preconditions),
            "predicted_transition": self.predicted_transition,
            "success_observation": self.success_observation,
            "abort_conditions": list(self.abort_conditions),
            "recovery": list(self.recovery),
            "risk_level": self.risk_level,
            "node_id": self.node_id,
            "geometry": list(self.geometry),
            "raw_text": self.raw_text,
            "created_at": self.created_at,
        }


@dataclass
class DecisionNeed:
    goal: str = ""
    unknowns: list[str] = field(default_factory=list)
    required_predicate: str = ""
    risk: str = RISK_MID
    minimum_evidence: str = "claim_or_keyframe"
    active_oid: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "unknowns": list(self.unknowns),
            "required_predicate": self.required_predicate,
            "risk": self.risk,
            "minimum_evidence": self.minimum_evidence,
            "active_oid": self.active_oid,
            "reason": self.reason,
        }


@dataclass
class VerifyResult:
    outcome: str                  # VERIFIED | FAILED | AMBIGUOUS
    reason: str = ""
    node_id: str = ""
    claim_updates: list[dict[str, Any]] = field(default_factory=list)
    physical_failure: str = ""    # drop | collision | misalignment | ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "node_id": self.node_id,
            "claim_updates": list(self.claim_updates),
            "physical_failure": self.physical_failure,
        }


@dataclass
class ArbiterDecision:
    verdict: str                  # act | recall | observe | retry | recover | hold | rewrite
    contract: ActionContract = field(default_factory=ActionContract)
    reason: str = ""
    reject_layer: str = ""        # which of the six layers rejected, if any
    rewritten: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "contract": self.contract.to_dict(),
            "reason": self.reason,
            "reject_layer": self.reject_layer,
            "rewritten": self.rewritten,
        }
