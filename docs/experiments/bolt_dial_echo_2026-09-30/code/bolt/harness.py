"""BoltHarness — the S_t = (b, G, L, E, q, c) controller.

Public API (BOLT.md §10):

    state    = harness.observe(obs, action_trace)
    need     = harness.compile_decision_need()
    context  = harness.context(need)
    decision = harness.arbitrate(proposal)
    result   = harness.verify(trace, decision.contract)
    harness.commit(result)
"""
from __future__ import annotations

import os
import time
from typing import Any

from .arbiter import UnifiedArbiter
from .board import render_board
from .claim import BeliefState
from .clock import SyncState
from .graph import ObligationGraph
from .ledger import VerifiedLedger
from .need import DecisionNeedCompiler
from .router import EvidenceRouter
from .types import (
    AV_ACT,
    AV_REWRITE,
    CS_COMMITTED,
    CS_OBSERVED,
    NS_ATTEMPTED,
    VR_AMBIGUOUS,
    VR_FAILED,
    VR_VERIFIED,
    ActionContract,
    ArbiterDecision,
    DecisionNeed,
    VerifyResult,
)
from .verify import IndependentVerifier

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def enabled() -> bool:
    return str(os.environ.get("MEMEXP_BOLT", "")).strip().lower() in _TRUTHY


class BoltHarness:
    """Fully-owned controller state. One instance per episode."""

    def __init__(self) -> None:
        self.belief = BeliefState()
        self.graph = ObligationGraph()
        self.ledger = VerifiedLedger()
        self.clock = SyncState()
        self.need_compiler = DecisionNeedCompiler()
        self.router = EvidenceRouter()
        self.arbiter = UnifiedArbiter()
        self.verifier = IndependentVerifier()

        self.last_need: DecisionNeed | None = None
        self.last_decision: ArbiterDecision | None = None
        self.last_board: str = ""
        self.last_verify: VerifyResult | None = None
        self.step: int = 0
        self.phi_curve: list[float] = []
        self.evidence_store: list[dict[str, Any]] = []  # Phase-2 event cards (stub)
        self.action_queue: list[ActionContract] = []
        self.retry_budget: int = int(os.environ.get("MEMEXP_BOLT_RETRY_BUDGET", "3") or 3)
        self.n_plan_steps = 0
        self.n_observe_calls = 0
        self.n_commit = 0
        self._verified_stages_seen: set[str] = set()
        self.seeded = False
        self.errors: list[str] = []

    # ------------------------------------------------------------------ seed -------------
    def seed(
        self,
        exec_labels: list[str],
        scored_names: list[str],
        label_to_scored: dict[str, str] | None = None,
        *,
        name_objects: bool = True,
    ) -> int:
        n = self.graph.seed_execution(
            exec_labels, scored_names, label_to_scored, name_objects=name_objects
        )
        self.seeded = n > 0
        self.phi_curve.append(self.graph.phi())
        # seed low-risk structural claims for open containers etc. as BELIEVED placeholders
        return n

    # ------------------------------------------------------------------ observe -----------
    def observe(self, obs: dict[str, Any] | None = None, action_trace: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fast-loop tick: update Claims/Ledger from obs + previous action receipt."""
        self.n_observe_calls += 1
        now = time.time()
        obs = obs or {}
        trace = action_trace or {}

        # progress guard: this planning decision is about to be directed at the
        # current obligation.  If the same node has been served `stall_limit` times
        # with Φ unchanged it cannot be closed and is demoted before the board is
        # rendered, so the planner is never told to repeat a dead instruction.
        act = self.graph.active()
        if act is not None:
            self.graph.note_serve(act.nid)

        # invalidation triggers from obs
        triggers = list(obs.get("invalidation_triggers") or [])
        if trace.get("object_moved"):
            triggers.append("object_moved")
        if trace.get("container_moved"):
            triggers.append("container_moved")
        if triggers:
            self.belief.check_invalidation_triggers(triggers, now=now)

        # harness-reported verified stages
        for st in obs.get("verified_stages") or []:
            self._ingest_verified_stage(str(st), step=self.step)

        if obs.get("new_evidence"):
            self.clock.mark_evidence(now)
            self.router.note_evidence_served()

        self.phi_curve.append(self.graph.phi())
        return self.snapshot()

    def note_verified_stages(self, stages: list[str] | set[str], *, step: int | None = None) -> list[str]:
        settled_all: list[str] = []
        for st in stages or []:
            settled_all.extend(self._ingest_verified_stage(str(st), step=step if step is not None else self.step))
        return settled_all

    def _ingest_verified_stage(self, stage_name: str, *, step: int) -> list[str]:
        if not stage_name or stage_name in self._verified_stages_seen:
            return []
        self._verified_stages_seen.add(stage_name)
        settled = self.graph.note_verified_stage(stage_name, step=step)
        for nid in settled:
            node = self.graph.nodes[nid]
            self.ledger.record_outcome(
                node_id=nid,
                outcome=VR_VERIFIED,
                reason=f"stage:{stage_name}",
                phase=node.phase,
                stage_name=stage_name,
            )
            if node.success_observation:
                self.belief.add(
                    node.success_observation,
                    status=CS_COMMITTED,
                    confidence=0.95,
                    source=[f"stage:{stage_name}"],
                    required_for=[nid],
                )
        self.clock.mark_evidence()
        return settled

    # ------------------------------------------------------------------ decision ----------
    def compile_decision_need(self) -> DecisionNeed:
        need = self.need_compiler.compile(self)
        self.last_need = need
        return need

    def context(self, need: DecisionNeed | None = None) -> str:
        need = need or self.last_need or self.compile_decision_need()
        board = render_board(self, need)
        self.last_board = board
        return board

    def arbitrate(self, proposal: ActionContract | str | None = None) -> ArbiterDecision:
        self.n_plan_steps += 1
        self.step += 1
        self.clock.mark_planner()
        need = self.last_need or self.compile_decision_need()
        route = self.router.route(self, need)
        decision = self.arbiter.arbitrate(self, proposal, need, router_mode=route["mode"])
        self.last_decision = decision
        # record attempt when we accept / rewrite to an actionable contract
        if decision.verdict in {AV_ACT, AV_REWRITE} and decision.contract.node_id:
            node = self.graph.nodes.get(decision.contract.node_id)
            self.graph.note_attempt(decision.contract.node_id)
            self.ledger.record_attempt(
                step=self.step,
                node_id=decision.contract.node_id,
                primitive=decision.contract.primitive,
                phase=node.phase if node else "",
            )
        return decision

    # ------------------------------------------------------------------ verify / commit ---
    def verify(
        self,
        action_trace: dict[str, Any] | None = None,
        contract: ActionContract | None = None,
        *,
        verified_stages: list[str] | None = None,
    ) -> VerifyResult:
        contract = contract or (self.last_decision.contract if self.last_decision else ActionContract())
        # merge any newly reported stages first
        if verified_stages:
            self.note_verified_stages(verified_stages)
        result = self.verifier.verify(
            self,
            contract,
            verified_stages=list(self._verified_stages_seen),
            action_trace=action_trace,
        )
        self.last_verify = result
        return result

    def commit(self, result: VerifyResult | None = None) -> None:
        result = result or self.last_verify
        if result is None:
            return
        self.n_commit += 1
        nid = result.node_id
        node = self.graph.nodes.get(nid) if nid else None

        if result.outcome == VR_VERIFIED:
            if nid and node and node.status != "SETTLED":
                # settle via verifies_stage if known, else mark settled directly
                if node.verifies_stage:
                    self.graph.note_verified_stage(node.verifies_stage, step=self.step)
                else:
                    node.status = "SETTLED"
                    node.settled_step = self.step
                self.ledger.record_outcome(
                    node_id=nid,
                    outcome=VR_VERIFIED,
                    reason=result.reason,
                    phase=node.phase if node else "",
                    stage_name=node.verifies_stage if node else "",
                )
            for upd in result.claim_updates:
                pred = upd.get("predicate") or ""
                if pred:
                    self.belief.add(
                        pred,
                        status=upd.get("status", CS_OBSERVED),
                        confidence=float(upd.get("confidence", 0.8)),
                        source=list(upd.get("source") or []),
                    )
        elif result.outcome == VR_FAILED:
            if nid:
                self.graph.note_failure(nid, reason=result.reason)
                self.ledger.record_outcome(
                    node_id=nid,
                    outcome=VR_FAILED,
                    reason=result.reason,
                    physical_failure=result.physical_failure,
                    phase=node.phase if node else "",
                )
            # drop confidence on related claims
            if node and node.success_observation:
                c = self.belief.find(node.success_observation)
                if c is not None:
                    self.belief.invalidate(c.claim_id, reason=result.reason)
        elif result.outcome == VR_AMBIGUOUS:
            if nid:
                self.graph.note_ambiguous(nid)
                self.ledger.record_outcome(
                    node_id=nid,
                    outcome=VR_AMBIGUOUS,
                    reason=result.reason,
                    phase=node.phase if node else "",
                )

        self.phi_curve.append(self.graph.phi())

    # ------------------------------------------------------------------ snapshot ----------
    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "seeded": self.seeded,
            "n_plan_steps": self.n_plan_steps,
            "n_observe_calls": self.n_observe_calls,
            "n_commit": self.n_commit,
            "phi": self.graph.phi(),
            "phi_curve_tail": self.phi_curve[-8:],
            "belief": self.belief.snapshot(),
            "graph": self.graph.snapshot(),
            "ledger": self.ledger.snapshot(),
            "clock": self.clock.snapshot(),
            "router": self.router.snapshot(),
            "arbiter": self.arbiter.snapshot(),
            "verifier": self.verifier.snapshot(),
            "last_need": self.last_need.to_dict() if self.last_need else None,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "last_verify": self.last_verify.to_dict() if self.last_verify else None,
            "errors": list(self.errors),
        }

    def report(self) -> dict[str, Any]:
        """Compact per-episode report for archival."""
        snap = self.snapshot()
        return {
            "totals": {
                "n_plan_steps": self.n_plan_steps,
                "n_observe_calls": self.n_observe_calls,
                "n_commit": self.n_commit,
                "phi_final": self.graph.phi(),
                "phi_initial": self.phi_curve[0] if self.phi_curve else None,
                "n_nodes": len(self.graph._order),
                "n_settled": sum(1 for n in self.graph.ordered() if n.status == "SETTLED"),
                "n_verified": self.ledger.n_verified,
                "n_failed": self.ledger.n_failed,
                "n_ambiguous": self.ledger.n_ambiguous,
                "n_accept": self.arbiter.n_accept,
                "n_rewrite": self.arbiter.n_rewrite,
                "n_hold": self.arbiter.n_hold,
                "n_clock_hold": self.arbiter.n_clock_hold,
                "n_dep_reject": self.arbiter.n_dep_reject,
                "n_recall": self.router.n_recall,
                "n_observe_route": self.router.n_observe,
                "n_floor_fire": self.router.n_floor_fire,
                "unsupported_claim_use": self.belief.n_unsupported_use,
                "stale_claim_use": self.belief.n_stale_use,
                "derived": self.ledger.snapshot()["derived"],
            },
            "coverage": self.graph.coverage(),
            "clock": snap["clock"],
            "exec_labels": list(self.graph.exec_labels),
            "scored_names": list(self.graph.scored_names),
            "errors": list(self.errors),
        }
