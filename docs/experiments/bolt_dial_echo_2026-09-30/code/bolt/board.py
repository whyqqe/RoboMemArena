"""Prompt board renderer — a pure function of (graph, ledger, belief, clock, need)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import BoltHarness
    from .types import DecisionNeed


DIGEST_CAP = 2400


def render_board(harness: "BoltHarness", need: "DecisionNeed | None" = None) -> str:
    g = harness.graph
    active = g.active()
    lines: list[str] = ["# BOLT-Sync obligation board"]

    # ACTIVE
    if active is None:
        lines.append("ACTIVE: (none — all settled or blocked)")
    else:
        # Serve the BDDL imperative (or its recovery rotation), never the symbolic
        # predicate. The predicate is written for graph reasoning ("grasp the tomato
        # and lift it clear of the source container"), and dumping it into the VLA
        # prompt was measured to leak verbatim into the planner output (73% of
        # task 19, 20% of task 8) and to degenerate into a noun-less "pour" when no
        # object was resolved.
        serve = g.serve_text(active)
        lines.append(f"ACTIVE: [{active.nid}] {active.phase} — {serve.rstrip('.')}")
        lines.append(f"  status={active.status} risk={active.risk_level} "
                     f"attempts={active.tau_act} fails={active.n_fail}"
                     f" serves={g._serve_counts.get(active.nid, 0)}"
                     f"{' [RECOVER]' if int(g._serve_counts.get(active.nid, 0)) >= 2 else ''}")
        lines.append(f"  serve: {serve}")
        # The previous command for this obligation is shown back with an explicit
        # do-not-repeat instruction.  Measured on v5 task 8, the failing trials were
        # not "the model chose the wrong step" — they were the model re-emitting one
        # sentence 14-17 times.  Naming that sentence in the context is the cheapest
        # way to force a genuinely different attempt, which is what recovers the VLA.
        prior = g.prev_served(active.nid)
        if prior and prior.strip() != serve.strip():
            lines.append(f"  previous attempt (do NOT repeat verbatim): {prior}")
        if active.deps:
            dep_bits = []
            for d, h in active.deps:
                dn = g.nodes.get(d)
                dep_bits.append(f"{dn.phase if dn else d}:{dn.status if dn else '?'}({h})")
            lines.append(f"  deps: {', '.join(dep_bits)}")

    # PROGRESS (derived counts from ledger ONLY)
    lines.append("PROGRESS:")
    for row in harness.ledger.derived_board():
        lines.append(f"  {row}")
    lines.append(f"  Φ(G) = {g.phi():.2f}")

    # OPEN ladder (compact).
    #
    # This line used to read `[n3] grasp tomato sauce [OPEN]` — i.e. a phase+object
    # pair, which is an *actionable, stage-ignoring robot command*.  The planner copied
    # it verbatim ("grasp tomato sauce bottle"), the grasp succeeded without a lift, the
    # node's scored stage (01_Lift_Tomato_Sauce) was never credited, and the episode was
    # spent re-issuing it: measured on v8 seeds 100/102/104, all 0.0, while the control
    # arm lifted in the same scenes.  Internal identifiers must never enter the action
    # channel, so the list names the SCORED STAGE instead, which is not executable.
    opens = [n for n in g.ordered() if n.is_open()]
    if opens:
        lines.append("OPEN:")
        for n in opens[:6]:
            mark = "*" if active and n.nid == active.nid else "-"
            tag = n.verifies_stage or n.settles_with or "unscored preparation"
            lines.append(f"  {mark} [{n.nid}] stage={tag} [{n.status}]")

    # Decision Need
    if need is not None:
        lines.append("NEED:")
        lines.append(f"  goal: {need.goal}")
        if need.unknowns:
            lines.append(f"  unknowns: {', '.join(need.unknowns)}")
        lines.append(f"  risk={need.risk}  min_evidence={need.minimum_evidence}")
        if need.reason:
            lines.append(f"  why: {need.reason}")

    # Sync
    ages = harness.clock.ages()
    def _fmt(v: float) -> str:
        return "inf" if v == float("inf") else f"{v:.1f}"
    lines.append(
        "SYNC: "
        f"cog={_fmt(ages['cognition_age'])}s "
        f"act={_fmt(ages['action_age'])}s "
        f"evi={_fmt(ages['evidence_age'])}s "
        f"gap={_fmt(ages['commitment_gap'])}s "
        f"window=[{ages['g_min']:.1f},{ages['g_max']:.1f}]"
    )

    # Claims (compact)
    live = [c for c in harness.belief.claims.values()
            if c.status not in {"STALE", "CONTRADICTED"}]
    if live:
        lines.append("CLAIMS:")
        for c in live[:5]:
            lines.append(f"  {c.predicate} = {c.value} [{c.status} conf={c.confidence:.2f}]")

    text = "\n".join(lines)
    if len(text) > DIGEST_CAP:
        text = text[:DIGEST_CAP] + "\n…(truncated)"
    return text
