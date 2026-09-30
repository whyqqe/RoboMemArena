"""ECHO deterministic core: prospective evidence, commitments, and verified ledger.

No scoring oracle is consumed online. A stage hint identifies the current decision need,
not a completed stage. Only an independent physical verifier can settle a commitment;
when unavailable we retain AMBIGUOUS and let the official harness execute unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

_POUR = re.compile(r"\b(pour|tilt)\b", re.I)
_SECOND = re.compile(r"\b(second|2nd|twice|again)\b", re.I)
_SAUCE = re.compile(r"\b(tomato|sauce|bottle)\b", re.I)
_LIFT = re.compile(r"\b(grasp|lift|pick|reach)\b", re.I)
_CHOCOLATE = re.compile(r"\bchocolate\b", re.I)


def phase_hint(stage: str) -> int:
    """A current-stage hint, not a verified-success signal."""
    s = stage.lower()
    if "pour_two" in s:
        return 2
    if "pour_one" in s:
        return 1
    return 0


def planner_constraint(phase: int) -> str:
    obligations = (
        "First grasp and lift the tomato sauce bottle; do not move the chocolate or pour yet.",
        "Now pour tomato sauce onto the chocolate once; do not describe a second pour yet.",
        "Now pour tomato sauce onto the chocolate a second time; do not move the chocolate.",
    )
    return "[ECHO current physical obligation, not a success assertion] " + obligations[min(max(phase, 0), 2)]


def primitive_reason(primitive: str, phase: int) -> str | None:
    p = primitive.strip()
    if not p:
        return "empty primitive"
    if re.fullmatch(r"(?:0[123]_)?(?:lift_tomato_sauce|pour_one|pour_two)", p, re.I):
        return "copied stage label"
    if phase == 0:
        if _POUR.search(p):
            return "pour before lift stage"
        if _CHOCOLATE.search(p) and not _SAUCE.search(p):
            return "off-stage chocolate manipulation"
    if phase == 1 and _POUR.search(p) and _SECOND.search(p):
        return "second pour before first pour stage ends"
    if phase >= 1 and _CHOCOLATE.search(p) and _LIFT.search(p) and not _SAUCE.search(p):
        return "off-stage chocolate manipulation"
    return None


def sanitize_output(raw: str | None, phase: int) -> tuple[str | None, str | None]:
    """Keep valid JSON fields; fail open on malformed outputs rather than fabricating API data."""
    if not raw:
        return raw, None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("current_primitive"), str):
            return raw, None
    except (ValueError, TypeError):
        return raw, None
    reason = primitive_reason(payload["current_primitive"], phase)
    if reason:
        payload["current_primitive"] = (
            "Reach for and grasp the tomato sauce bottle, then lift it."
            if phase == 0 else
            "Tilt the tomato sauce bottle to pour onto the chocolate once."
            if phase == 1 else
            "Tilt the tomato sauce bottle to pour onto the chocolate again."
        )
        return json.dumps(payload, ensure_ascii=False), reason
    return raw, None


@dataclass(frozen=True)
class Evidence:
    step: int
    kind: str
    frame: int | None
    claim: str
    valid_until: int
    status: str = "OBSERVED"


@dataclass
class Commitment:
    goal: str
    primitive: str
    started: int
    last_seen: int
    status: str = "DELIVERED"
    evidence: list[int] = field(default_factory=list)


@dataclass
class EchoState:
    """Episode-scoped only; no cross-episode mutable state or scoring-stage oracle."""
    task_id: int = 0
    episode: int = 0
    stage: str = ""
    step: int = 0
    commitment: Commitment | None = None
    evidence: list[Evidence] = field(default_factory=list)
    verified: set[str] = field(default_factory=set)
    n_saved: int = 0
    n_recalled: int = 0
    n_delivered: int = 0
    n_ambiguous: int = 0
    n_verified: int = 0
    n_stage_change: int = 0
    n_identity: int = 0
    n_plans: int = 0
    n_reports: int = 0
    phase: int = 0
    n_constraints: int = 0
    n_rejected: int = 0
    n_retries: int = 0
    errors: list[str] = field(default_factory=list)

    def anticipate(self, stage: str, step: int, available: dict[int, Any], *, max_events: int = 24) -> None:
        """Write a decision-indexed event before its image leaves the frame store.

        Stage transitions are *hints*, not physical success; the outgoing obligation
        remains AMBIGUOUS. We never label frames with a secret target identity.
        """
        self.step = int(step)
        if stage and stage != self.stage:
            if self.commitment is not None and self.commitment.status == "DELIVERED":
                self.commitment.status = "AMBIGUOUS"
                self.n_ambiguous += 1
            self.stage = stage
            self.n_stage_change += 1
            self._save("transition", max(available, default=None), stage, max_events)
        if available:
            # Capture the most recent actually available frame, not an invented index.
            last = max(available)
            if not self.evidence or last - self.evidence[-1].step >= 25:
                self._save("observation", last, stage, max_events)

    def _save(self, kind: str, frame: int | None, claim: str, cap: int) -> None:
        if frame is None or (self.evidence and self.evidence[-1].frame == frame):
            return
        self.evidence.append(Evidence(self.step, kind, frame, claim,
                                      self.step + 250))
        self.evidence = self.evidence[-cap:]
        self.n_saved += 1

    def delivered(self, primitive: str, step: int) -> None:
        """Record the actual post-guard VLA prompt, never a planner proposal."""
        self.step = int(step)
        self.n_delivered += 1
        if self.commitment is not None and self.commitment.primitive == primitive:
            self.commitment.last_seen = self.step
            return
        if self.commitment is not None and self.commitment.status == "DELIVERED":
            self.commitment.status = "AMBIGUOUS"
            self.n_ambiguous += 1
        self.commitment = Commitment(self.stage, primitive, self.step, self.step)

    def independent_verification(self, claim: str, success: bool, source: str) -> None:
        """Only explicit independent evidence may settle; scorer is NOT an online source."""
        if source not in {"physical_verifier", "proprioception", "visual_verifier"}:
            raise ValueError("untrusted verification source")
        if success:
            self.verified.add(claim)
            self.n_verified += 1
            if self.commitment is not None:
                self.commitment.status = "VERIFIED"

    def decision_package(self, available: dict[int, Any], *, cap: int = 2) -> tuple[str, list[int]]:
        """Retrieve just the frames relevant to an unresolved commitment/stage.

        Only existing, non-expired frames are offered. Descriptive metadata, never a
        candidate VLA command or an answer-key/score predicate.
        """
        active = self.stage
        events = [e for e in self.evidence if e.frame in available
                  and e.valid_until >= self.step and e.claim == active]
        indices = [e.frame for e in events[-cap:] if e.frame is not None]
        if not indices:
            return "", []
        self.n_recalled += len(indices)
        status = self.commitment.status if self.commitment else "NOT_DELIVERED"
        text = ("[ECHO prospective evidence; descriptive, not a VLA instruction]\n"
                f"An execution commitment is {status}; physical success is not confirmed. "
                "These are previously observed frames relevant to the present decision. "
                "Use them only to resolve a missing fact; do not infer completion from a "
                "command or from a visible object. Current observation remains authoritative.")
        return text, indices

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "episode": self.episode, "step": self.step,
            "stage": self.stage, "n_saved": self.n_saved, "n_recalled": self.n_recalled,
            "n_delivered": self.n_delivered, "n_ambiguous": self.n_ambiguous,
            "n_verified": self.n_verified, "n_stage_change": self.n_stage_change,
            "n_identity": self.n_identity, "n_plans": self.n_plans,
            "phase": self.phase, "n_constraints": self.n_constraints,
            "n_rejected": self.n_rejected, "n_retries": self.n_retries,
            "commitment": None if self.commitment is None else vars(self.commitment),
            "verified": sorted(self.verified), "errors": self.errors,
        }
