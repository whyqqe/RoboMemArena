"""Decision Need Compiler — structured gaps, not model introspection.

The Decision Need is compiled from (belief, graph, ledger, clock) state. The
planner never has to "notice that it is missing something".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import RISK_HIGH, RISK_MID, DecisionNeed

if TYPE_CHECKING:
    from .harness import BoltHarness


class DecisionNeedCompiler:
    def compile(self, harness: "BoltHarness") -> DecisionNeed:
        g = harness.graph
        active = g.active()
        unknowns: list[str] = []
        reason_parts: list[str] = []

        if active is None:
            return DecisionNeed(goal="idle", reason="no open obligation", risk=RISK_MID)

        # missing preconditions on the active node → unknowns
        for dep_id, hardness in active.deps:
            dep = g.nodes.get(dep_id)
            if dep is None:
                continue
            if not g.deps_ready(active):
                unknowns.append(f"dep:{dep.primitive}")
                reason_parts.append(f"unmet {hardness} dep {dep.phase}")

        # needs_look without evidence
        if active.needs_look and active.status == "OPEN" and active.tau_act == 0:
            unknowns.append(f"look:{active.primitive}")
            reason_parts.append("needs_look before act")

        # recent physical failures on this node → inspect
        if active.n_fail > 0:
            unknowns.append(f"failure_mode:{active.nid}")
            reason_parts.append(f"prior failures={active.n_fail}")

        # dual-clock pressure
        ok, clock_why = harness.clock.allow_commit(world_uncertain=bool(unknowns))
        if not ok:
            unknowns.append("sync:commitment_gap")
            reason_parts.append(clock_why)

        risk = active.risk_level or RISK_MID
        if unknowns and risk != RISK_HIGH:
            risk = RISK_HIGH if any(u.startswith("dep:") for u in unknowns) else risk

        goal = f"advance:{active.phase}"
        if active.object:
            goal += f":{active.object}"

        min_ev = "current_view" if active.needs_look else "claim_or_keyframe"
        if active.n_fail > 0:
            min_ev = "keyframe_or_current_view"

        return DecisionNeed(
            goal=goal,
            unknowns=unknowns,
            required_predicate=active.predicate or active.success_observation,
            risk=risk,
            minimum_evidence=min_ev,
            active_oid=active.nid,
            reason="; ".join(reason_parts) or "advance active obligation",
        )
