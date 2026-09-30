"""Unified Arbiter — six-layer check over memory, action, sync and safety.

Layers (in order):
  1. graph legality
  2. Claim preconditions
  3. dual-clock consistency
  4. geometry (Phase-3 stub; passes when MEMEXP_BOLT_GEOMETRY=0)
  5. safety STL (Phase-4 stub; passes when MEMEXP_BOLT_SAFETY=0)
  6. progress / risk (Φ must not rise from a known-bad repeat)

On reject: rewrite to the ACTIVE obligation's first admissible template
(corrective command, never a no-op).
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from . import serve as S
from .types import (
    AV_ACT,
    AV_HOLD,
    AV_OBSERVE,
    AV_RECALL,
    AV_RECOVER,
    AV_REWRITE,
    ActionContract,
    ArbiterDecision,
    DecisionNeed,
)

if TYPE_CHECKING:
    from .harness import BoltHarness

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in _TRUTHY


class UnifiedArbiter:
    def __init__(self) -> None:
        self.n_accept = 0
        self.n_rewrite = 0
        self.n_hold = 0
        self.n_reject = 0
        self.reject_by_layer: dict[str, int] = {}
        self.n_gate_reject = 0
        self.n_dep_reject = 0
        self.n_ordinal_reject = 0
        self.n_verbatim_blocks = 0
        self.n_stage_reject = 0
        self.n_offgraph = 0
        self.n_off_active = 0
        self.n_clock_hold = 0
        self.n_claim_reject = 0
        self.n_geom_reject = 0
        self.n_safety_reject = 0
        self.n_progress_reject = 0

    def arbitrate(
        self,
        harness: "BoltHarness",
        proposal: ActionContract | str | None,
        need: DecisionNeed,
        *,
        router_mode: str = AV_ACT,
    ) -> ArbiterDecision:
        # Router may already demand recall / observe / hold
        if router_mode == AV_RECALL:
            return ArbiterDecision(verdict=AV_RECALL, reason="router:recall", contract=self._as_contract(proposal))
        if router_mode == AV_OBSERVE:
            return ArbiterDecision(verdict=AV_OBSERVE, reason="router:observe", contract=self._as_contract(proposal))

        contract = self._as_contract(proposal)
        if not contract.primitive.strip():
            # empty proposal → rewrite to ACTIVE
            return self._rewrite(harness, reason="empty primitive", layer="graph")

        # bind to a graph node if possible
        matches = harness.graph.match_primitive(contract.primitive)
        if matches:
            contract.node_id = matches[0].nid
            contract.risk_level = matches[0].risk_level
            if not contract.object:
                contract.object = matches[0].object
            if not contract.target:
                contract.target = matches[0].target
            if not contract.preconditions:
                contract.preconditions = [
                    f"dep_ready:{d}" for d, _ in matches[0].deps
                ]
            if not contract.success_observation:
                contract.success_observation = matches[0].success_observation
            if not contract.recovery:
                contract.recovery = list(matches[0].recovery_edges)

        # --- layer 1: graph legality ------------------------------------------------------
        ok, why = self._check_graph(harness, contract)
        if not ok:
            return self._rewrite(harness, reason=why, layer="graph")

        # --- layer 2: Claim preconditions -------------------------------------------------
        ok, why = self._check_claims(harness, contract)
        if not ok:
            return self._rewrite(harness, reason=why, layer="claim")

        # --- layer 3: dual-clock ----------------------------------------------------------
        ok, why = harness.clock.allow_commit(world_uncertain=bool(need.unknowns))
        if not ok:
            self.n_clock_hold += 1
            self.n_hold += 1
            self.reject_by_layer["clock"] = self.reject_by_layer.get("clock", 0) + 1
            verdict = AV_OBSERVE if "observe" in why else AV_HOLD
            return ArbiterDecision(
                verdict=verdict,
                contract=contract,
                reason=why,
                reject_layer="clock",
            )

        # --- layer 4: geometry (stub) -----------------------------------------------------
        if _flag("MEMEXP_BOLT_GEOMETRY", "0"):
            ok, why = self._check_geometry(contract)
            if not ok:
                self.n_geom_reject += 1
                return self._rewrite(harness, reason=why, layer="geometry")

        # --- layer 5: safety STL (stub) ---------------------------------------------------
        if _flag("MEMEXP_BOLT_SAFETY", "0"):
            ok, why = self._check_safety(contract)
            if not ok:
                self.n_safety_reject += 1
                return ArbiterDecision(
                    verdict=AV_RECOVER,
                    contract=contract,
                    reason=why,
                    reject_layer="safety",
                )

        # --- layer 6: progress / risk ------------------------------------------------------
        ok, why = self._check_progress(harness, contract)
        if not ok:
            self.n_progress_reject += 1
            return self._rewrite(harness, reason=why, layer="progress")

        contract.created_at = time.time()
        harness.clock.mark_contract(contract.created_at)
        self.n_accept += 1
        return ArbiterDecision(verdict=AV_ACT, contract=contract, reason="accepted")

    # -- layer implementations -------------------------------------------------------------
    def _check_graph(self, harness: "BoltHarness", contract: ActionContract) -> tuple[bool, str]:
        if not contract.node_id:
            # unmatched paraphrase — conservative allow (AOM lesson), but count it
            self.n_offgraph += 1
            return True, ""
        node = harness.graph.nodes.get(contract.node_id)
        if node is None:
            self.n_gate_reject += 1
            return False, "node missing from graph"
        if not harness.graph.deps_ready(node):
            self.n_dep_reject += 1
            unmet = []
            ordinal_blocked = False
            for d, h in node.deps:
                dep = harness.graph.nodes.get(d)
                if dep is None:
                    continue
                served = int(harness.graph._serve_counts.get(dep.nid, 0)) > 0
                soft_ok = h == "soft" and (
                    dep.status != "OPEN" or int(dep.tau_act) > 0 or served
                )
                hard_ok = h == "hard" and dep.status == "SETTLED"
                if not (soft_ok or hard_ok):
                    unmet.append(f"{dep.phase}[{dep.status}]")
                    # I4: the most consequential hard gate is pour_two ⇐ pour_one.
                    # The VLA will happily emit "pour ... a second time" while the
                    # first pour is still open; the graph must refuse it.
                    if h == "hard" and dep.phase.startswith("pour"):
                        ordinal_blocked = True
            if ordinal_blocked:
                self.n_ordinal_reject += 1
            return False, f"deps not ready: {','.join(unmet) or 'unknown'}"
        if node.status == "SETTLED":
            self.n_gate_reject += 1
            return False, f"node {node.nid} already SETTLED"
        # Obligation discipline: acting on some other open obligation is how a
        # planner escapes the graph.  Without this check `_check_graph` accepted any
        # ready node, so on task 8 the planner could keep answering "grasp chocolate"
        # while the board was directing it at "place chocolate in frypan" -- the
        # board had no effect and the ladder never advanced.  The graph is an
        # obligation *order*, so only the ACTIVE obligation is admissible.
        active = harness.graph.active()
        if active is not None and active.nid != node.nid:
            self.n_off_active += 1
            return False, (
                f"proposal on {node.nid}({node.phase}) but ACTIVE is "
                f"{active.nid}({active.phase})"
            )
        return True, ""

    def _check_claims(self, harness: "BoltHarness", contract: ActionContract) -> tuple[bool, str]:
        # Only enforce Claim predicates that look like real Claim keys (held(...), open(...)).
        preds = [p for p in (contract.preconditions or []) if "(" in p and not p.startswith("dep_ready:")]
        if not preds:
            return True, ""
        ok, why = harness.belief.precondition_ok(preds, risk=contract.risk_level)
        if not ok:
            self.n_claim_reject += 1
        return ok, why

    def _check_geometry(self, contract: ActionContract) -> tuple[bool, str]:
        # Phase-3 stub: if geometry constraints are listed, require them to be non-vacuous.
        if contract.geometry and any("unsat" in g.lower() for g in contract.geometry):
            return False, "geometry unsatisfiable"
        return True, ""

    def _check_safety(self, contract: ActionContract) -> tuple[bool, str]:
        # Phase-4 stub: reject explicitly unsafe abort tags when safety module is on.
        if any(a in {"self_collision", "human_proximity"} for a in contract.abort_conditions):
            return False, "safety predicate flagged"
        return True, ""

    def _check_progress(self, harness: "BoltHarness", contract: ActionContract) -> tuple[bool, str]:
        g = harness.graph
        if not contract.node_id:
            return True, ""
        node = g.nodes.get(contract.node_id)
        if node is None:
            return True, ""
        # reject repeating a mode that just failed without recovery
        if node.n_fail >= 2 and not contract.recovery:
            return False, f"repeated failure on {node.nid} without recovery plan"

        # ---- stage satisfiability (I5, enforced at runtime) ------------------------------
        # A proposal that cannot satisfy the stage this node is verified by is not a weak
        # attempt, it is a guaranteed dead step: the environment has nothing to credit, so
        # Φ never drops and this same node stays ACTIVE while the episode is spent on it.
        # Measured on v8 task 8: the board's OPEN list exposed `grasp tomato sauce`, the
        # planner copied it, and because a grasp succeeds while the scored stage is the
        # LIFT, seeds 100/102/104 all ran to timeout at 0.0 while the control arm lifted in
        # the same scenes.  Enforced here rather than only asserted at build time so that no
        # future leak — from any source — can reach the environment as a dead instruction.
        if not S.satisfies_stage(node.phase, contract.primitive):
            self.n_stage_reject += 1
            return False, (
                f"{node.nid}({node.phase}) cannot be satisfied by "
                f"{contract.primitive!r} — needs a lift/place/pour verb"
            )

        # ---- variation discipline --------------------------------------------------------
        # Forbid the byte-identical repetition of the command just issued for this
        # obligation.  A VLA is a policy, not a planner: the same sentence reproduces
        # the same (failed) trajectory, and the archive shows this is exactly how the
        # scored stages were lost — v5 task 8 served one sentence 14/17, 15/17, 16/18
        # and 17/19 times in the four trials that never lifted, while the control arm
        # rephrased the same intent every step and lifted in the same scenes.
        # Any *other* legal phrasing is accepted: variation is the objective, and the
        # graph-legality layer has already bound the proposal to the active obligation.
        prev = self._norm(g.prev_served(node.nid))
        got = self._norm(contract.primitive)
        if got and prev and got == prev:
            self.n_verbatim_blocks += 1
            g.n_verbatim_blocks = int(getattr(g, "n_verbatim_blocks", 0)) + 1
            return False, f"verbatim repeat on {node.nid} → {g.serve_text(node)!r}"
        return True, ""

    @staticmethod
    def _norm(text: str) -> str:
        return str(text or "").strip().rstrip(".").lower()

    # -- rewrite ---------------------------------------------------------------------------
    def _rewrite(self, harness: "BoltHarness", *, reason: str, layer: str) -> ArbiterDecision:
        self.n_reject += 1
        self.n_rewrite += 1
        self.reject_by_layer[layer] = self.reject_by_layer.get(layer, 0) + 1
        active = harness.graph.active()
        if active is None:
            return ArbiterDecision(
                verdict=AV_HOLD,
                reason=f"reject[{layer}]: {reason}; no active node to rewrite to",
                reject_layer=layer,
            )
        tmpl = harness.graph.serve_text(active)
        contract = ActionContract(
            primitive=tmpl,
            object=active.object,
            target=active.target,
            node_id=active.nid,
            risk_level=active.risk_level,
            success_observation=active.success_observation,
            recovery=list(active.recovery_edges),
            created_at=time.time(),
            raw_text=tmpl,
        )
        harness.clock.mark_contract(contract.created_at)
        return ArbiterDecision(
            verdict=AV_REWRITE,
            contract=contract,
            reason=f"reject[{layer}]: {reason} → rewrite to {active.nid}:{active.phase}",
            reject_layer=layer,
            rewritten=True,
        )

    def _as_contract(self, proposal: ActionContract | str | None) -> ActionContract:
        if isinstance(proposal, ActionContract):
            return proposal
        text = str(proposal or "").strip()
        return ActionContract(primitive=text, raw_text=text)

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_accept": self.n_accept,
            "n_rewrite": self.n_rewrite,
            "n_hold": self.n_hold,
            "n_reject": self.n_reject,
            "reject_by_layer": dict(self.reject_by_layer),
            "n_gate_reject": self.n_gate_reject,
            "n_dep_reject": self.n_dep_reject,
            "n_ordinal_reject": self.n_ordinal_reject,
            "n_verbatim_blocks": self.n_verbatim_blocks,
            "n_stage_reject": self.n_stage_reject,
            "n_offgraph": self.n_offgraph,
            "n_off_active": self.n_off_active,
            "n_clock_hold": self.n_clock_hold,
            "n_claim_reject": self.n_claim_reject,
            "n_geom_reject": self.n_geom_reject,
            "n_safety_reject": self.n_safety_reject,
            "n_progress_reject": self.n_progress_reject,
        }
