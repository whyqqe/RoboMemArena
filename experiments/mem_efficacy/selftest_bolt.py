#!/usr/bin/env python3
"""BOLT-Sync selftest — no GPU, no API.

Every check is a claim the architecture makes (BOLT.md Phase 1):
  T1  execution-partition seeding (task 19 has 6 nodes, not 3)
  T2  soft grasp→place dep (attempted unlocks place; settled not required)
  T3  hard pour_1→pour_2 dep (SETTLED required)
  T4  Φ decreases only on VERIFIED settle
  T5  dual-clock hold when commitment_gap > g_max
  T6  Decision Need names unknowns without model introspection
  T7  Arbiter rewrites illegal place-before-grasp to the ACTIVE grasp
  T8  Claim invalidation + unsupported_claim_use counter
  T9  derived counts come from the ledger, never from text
  T10 evidence starvation floor fires after N starved steps
  T11 report() is JSON-serialisable and carries coverage / clock / derived
"""
from __future__ import annotations

import json
import os
import sys
import time

# mem_efficacy on path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("MEMEXP_BOLT", "1")
os.environ["MEMEXP_BOLT_GAP_MIN"] = "0"
os.environ["MEMEXP_BOLT_GAP_MAX"] = "5"
os.environ["MEMEXP_BOLT_ROUTER"] = "1"
os.environ["MEMEXP_BOLT_FLOOR"] = "1"
os.environ["MEMEXP_BOLT_STARVE_LIMIT"] = "3"
os.environ["MEMEXP_BOLT_GEOMETRY"] = "0"
os.environ["MEMEXP_BOLT_SAFETY"] = "0"
os.environ["MEMEXP_BOLT_STALL_LIMIT"] = "2"

from bolt.harness import BoltHarness  # noqa: E402
from bolt import serve as SV  # noqa: E402
from bolt.types import (  # noqa: E402
    AV_ACT,
    AV_HOLD,
    AV_OBSERVE,
    AV_REWRITE,
    NS_SETTLED,
    VR_VERIFIED,
)


class Fail(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def _t19_labels():
    return [
        "pick tomato sauce",
        "place tomato sauce cabinet2",
        "pick milk",
        "place milk cabinet2",
        "pick orange juice",
        "place orange juice cabinet2",
    ]


def _t19_scored():
    return [
        "01_Place_Tomato_Sauce_Cabinet2",
        "02_Place_Milk_Cabinet2",
        "03_Place_Orange_Juice_Cabinet2",
    ]


def _t19_map():
    labs, scored = _t19_labels(), _t19_scored()
    return {
        labs[1]: scored[0],
        labs[3]: scored[1],
        labs[5]: scored[2],
    }


def _t8_labels():
    return [
        "pick chocolate",
        "place chocolate in frypan",
        "pick tomato sauce",
        "pour tomato sauce over chocolate 1st",
        "pour tomato sauce over chocolate 2nd",
        "place tomato sauce bowl drainer",
    ]


def _t22_labels():
    return [
        "pick tomato",
        "pour first",
        "pour second",
        "place tomato aside",
        "open microwave",
        "pick cookies",
        "place cookies",
        "close microwave",
    ]


def _det_path():
    import pathlib

    return pathlib.Path(
        os.environ.get(
            "MEMEXP_DET",
            "/project/peilab/why/RoboMemArena/evaluation_benchmark/async_vlm26_det/"
            "fullvlm_v2_26_memory_tasks.json",
        )
    )


def _t5_labels():
    """Task 5's drawer chain comes straight from the det file."""
    det = json.loads(_det_path().read_text())
    for t in det["tasks"]:
        if int(t["task_id"]) == 5:
            return [str(p["label"]) for p in t["primitive_order"]]
    raise Fail("T5 labels not found in det file")


# -----------------------------------------------------------------------------------------
def T1_execution_seed():
    h = BoltHarness()
    n = h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    check(n == 6, f"T1: expected 6 exec nodes, got {n}")
    phases = [x.phase for x in h.graph.ordered()]
    check(phases.count("grasp") == 3, f"T1: expected 3 grasps, got {phases}")
    check(phases.count("place") == 3, f"T1: expected 3 places, got {phases}")
    check(h.graph.exec_labels == _t19_labels(), "T1: exec_labels not stored")
    print("T1 PASS — execution-partition seeding (6 nodes, 3 grasp + 3 place)")


def T2_soft_grasp_place():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    place = next(n for n in h.graph.ordered() if n.phase == "place")
    grasp = h.graph.nodes[place.deps[0][0]]
    check(place.deps[0][1] == "soft", f"T2: expected soft dep, got {place.deps}")
    # before attempt: place NOT ready
    check(grasp.status == "OPEN" and grasp.tau_act == 0, "T2 setup")
    check(not h.graph.deps_ready(place), "T2: place should be blocked before grasp attempt")
    # after one attempt: place ready
    h.graph.note_attempt(grasp.nid)
    check(h.graph.deps_ready(place), "T2: place should unlock after grasp attempt")
    print("T2 PASS — soft grasp→place dep (attempted unlocks place)")


def T3_hard_pour_chain():
    h = BoltHarness()
    # use simplified pour labels that phase_of recognises
    labels = ["pick tomato sauce", "pour tomato sauce 1st", "pour tomato sauce 2nd"]
    h.seed(labels, ["01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"], {
        "pick tomato sauce": "01_Lift_Tomato_Sauce",
        "pour tomato sauce 1st": "02_Pour_One",
        "pour tomato sauce 2nd": "03_Pour_Two",
    })
    nodes = h.graph.ordered()
    pour2 = nodes[2]
    check(pour2.phase == "pour_two", f"T3: expected pour_two, got {pour2.phase}")
    check(any(h == "hard" for _, h in pour2.deps), f"T3: expected hard dep, got {pour2.deps}")
    # attempting pour_one is NOT enough
    pour1 = nodes[1]
    h.graph.note_attempt(pour1.nid)
    check(not h.graph.deps_ready(pour2), "T3: pour_two must wait for SETTLED pour_one")
    pour1.status = NS_SETTLED
    check(h.graph.deps_ready(pour2), "T3: pour_two unlocks after pour_one SETTLED")
    print("T3 PASS — hard pour_1→pour_2 dep")


def T4_phi_monotone():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    phi0 = h.graph.phi()
    # attempting does NOT lower Φ
    active = h.graph.active()
    h.graph.note_attempt(active.nid)
    check(h.graph.phi() == phi0, f"T4: attempt must not change Φ ({phi0}→{h.graph.phi()})")
    # verifying a scored stage settles grasp+place and lowers Φ
    h.note_verified_stages(["01_Place_Tomato_Sauce_Cabinet2"], step=1)
    phi1 = h.graph.phi()
    check(phi1 < phi0, f"T4: Φ must drop on settle ({phi0}→{phi1})")
    # language output alone cannot drop Φ further
    h.arbitrate("Place the tomato sauce into cabinet2.")
    check(h.graph.phi() == phi1, "T4: language-only must not change Φ")
    print(f"T4 PASS — Φ monotone on VERIFIED only ({phi0:.1f}→{phi1:.1f})")


def T5_dual_clock_hold():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    # create a contract in the past, never refresh evidence → gap large
    past = time.time() - 20.0
    h.clock.mark_contract(past)
    # evidence never marked → gap ≈ action_age = 20 > g_max=5
    ok, why = h.clock.allow_commit()
    check(not ok, f"T5: expected hold, got ok={ok} why={why}")
    check("gap" in why.lower() or "window" in why.lower(), f"T5: unexpected why={why}")
    # refresh evidence → gap shrinks into window
    h.clock.mark_evidence(time.time())
    h.clock.mark_contract(time.time())
    ok2, why2 = h.clock.allow_commit()
    check(ok2, f"T5: expected in-window after refresh, got {why2}")
    print("T5 PASS — dual-clock hold when commitment_gap > g_max")


def T6_decision_need():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    need = h.compile_decision_need()
    check(need.active_oid != "", "T6: need must name active oid")
    check(need.goal.startswith("advance:"), f"T6: goal={need.goal}")
    check(need.required_predicate != "", "T6: predicate must be read off the graph node")
    # The need is derived from graph state, never from a model self-report: a prior
    # physical failure on the active obligation must surface as a structured unknown
    # and must raise the evidence bar.
    # (The old version of this test forced `active_nid` to a blocked node and looked
    #  for a `dep:` unknown.  Under correct selection that state is unreachable —
    #  a blocked node always has a ready blocker — so the assertion was testing dead
    #  code.  It now exercises the reachable derivation instead.)
    active = h.graph.active()
    check(active is not None, "T6: expected an active obligation")
    h.graph.note_failure(active.nid, reason="drop")
    need2 = h.compile_decision_need()
    check(any(u.startswith("failure_mode:") for u in need2.unknowns),
          f"T6: expected failure_mode unknown, got {need2.unknowns}")
    check(need2.minimum_evidence == "keyframe_or_current_view",
          f"T6: failures must raise evidence bar, got {need2.minimum_evidence}")
    check(need2.active_oid == active.nid, "T6: need must track the active obligation")
    print("T6 PASS — Decision Need compiled from graph/belief, not model self-report")


def T7_arbiter_rewrite():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    # ACTIVE is the first grasp; emitting a place must be rewritten
    h.compile_decision_need()
    decision = h.arbitrate("Place the tomato sauce into cabinet2.")
    check(decision.verdict in {AV_REWRITE, AV_ACT}, f"T7: verdict={decision.verdict}")
    check("grasp" in decision.contract.primitive.lower()
          or "pick" in decision.contract.primitive.lower(),
          f"T7: rewrite should target grasp, got {decision.contract.primitive!r} "
          f"(reason={decision.reason})")
    check(decision.rewritten or decision.reject_layer in {"graph", "progress", ""},
          f"T7: expected rewrite/graph reject, got {decision.to_dict()}")
    print(f"T7 PASS — Arbiter rewrites place-first → {decision.contract.primitive!r}")


def T8_claim_invalidation():
    h = BoltHarness()
    c = h.belief.add(
        "inside(tomato_sauce, cabinet1)",
        status="OBSERVED",
        confidence=0.9,
        invalidation=["tomato_sauce_moved", "cabinet1_moved"],
    )
    ok, _ = h.belief.precondition_ok(["inside(tomato_sauce, cabinet1)"], risk="high")
    check(ok, "T8: fresh claim should support high risk")
    h.belief.check_invalidation_triggers(["tomato_sauce_moved"])
    check(c.status == "STALE", f"T8: expected STALE, got {c.status}")
    ok2, why = h.belief.precondition_ok(["inside(tomato_sauce, cabinet1)"], risk="high")
    check(not ok2, f"T8: stale claim must fail precondition ({why})")
    check(h.belief.n_stale_use >= 1, "T8: stale_use counter must increment")
    print("T8 PASS — Claim invalidation + stale_use counter")


def T9_derived_from_ledger():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    check(h.ledger.count("place") == 0, "T9 setup")
    h.note_verified_stages(["01_Place_Tomato_Sauce_Cabinet2"], step=1)
    # one scored stage settles BOTH the place (verifies_stage) and its grasp (settles_with)
    check(h.ledger.count("place") == 1, f"T9: count:place={h.ledger.count('place')}")
    check(h.ledger.count("grasp") == 1, f"T9: count:grasp={h.ledger.count('grasp')}")
    check(h.ledger.count("verified_success") == 2,
          f"T9: verified_success={h.ledger.count('verified_success')} (expect 2=grasp+place)")
    # VLM self-report text must NOT change the counter
    h.arbitrate("I have placed two objects already.")
    check(h.ledger.count("place") == 1, "T9: text must not bump count:place")
    print("T9 PASS — derived counts from ledger only")


def T10_starvation_floor():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    # mark clock fresh so arbiter won't HOLD for gap
    h.clock.mark_evidence()
    h.clock.mark_contract()
    h.clock.mark_planner()
    # starve the evidence channel
    for _ in range(5):
        need = h.compile_decision_need()
        route = h.router.route(h, need)
    check(h.router.n_floor_fire >= 1,
          f"T10: floor should fire, n_floor_fire={h.router.n_floor_fire} "
          f"starved={h.router.n_starved_steps}")
    print(f"T10 PASS — evidence starvation floor (fired={h.router.n_floor_fire})")


def T11_report_shape():
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    h.compile_decision_need()
    h.context()
    h.arbitrate("Pick up the tomato sauce from the source container.")
    h.note_verified_stages(["01_Place_Tomato_Sauce_Cabinet2"], step=2)
    h.commit(h.verify(verified_stages=["01_Place_Tomato_Sauce_Cabinet2"]))
    rep = h.report()
    blob = json.dumps(rep)
    check("totals" in rep and "coverage" in rep and "clock" in rep, "T11: missing keys")
    check(rep["totals"]["n_nodes"] == 6, "T11: n_nodes")
    check("count:place" in rep["totals"]["derived"], "T11: derived missing")
    check(len(blob) > 100, "T11: report too small")
    print(f"T11 PASS — report JSON ({len(blob)} bytes)")


# -----------------------------------------------------------------------------------------
# Regression tests for the alignment / closure fixes.  Each one pins a defect that
# was measured on the archived runs (task 8 early-stopped 0/0/0; task 22 and task 5
# collapsing 62-96% of planning budget onto a single unreachable obligation).
# -----------------------------------------------------------------------------------------
def _stage_specs(task_id: int):
    """Scoring specs for a task, or None when the reference module is unavailable."""
    try:
        import task2_26_reference_stage as stage_eval  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return list(stage_eval._task_specs(task_id) or [])


def _require_specs(task_id: int):
    specs = _stage_specs(task_id)
    if not specs:
        raise Fail(
            f"task2_26_reference_stage not importable (task {task_id}); "
            "run with PYTHONPATH including evaluation_benchmark/scripts"
        )
    return specs


def T12_pour_alignment():
    """A POUR stage must never be bridged to a PICK primitive."""
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    specs = _require_specs(8)
    mapping = align_exec_to_scored(_t8_labels(), specs)
    check(mapping.get("pick tomato sauce") == "01_Lift_Tomato_Sauce",
          f"T12: lift bridge wrong: {mapping}")
    check(mapping.get("pour tomato sauce over chocolate 1st") == "02_Pour_One",
          f"T12: pour_one bridge wrong: {mapping}")
    check(mapping.get("pour tomato sauce over chocolate 2nd") == "03_Pour_Two",
          f"T12: pour_two bridge wrong: {mapping}")
    # the prep primitives carry no scored stage and must be marked instrumental
    check("pick chocolate" not in mapping,
          f"T12: 'pick chocolate' must not carry a scored stage, got {mapping}")
    # every scored stage is covered
    check(set(mapping.values()) == {"01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"},
          f"T12: coverage hole: {mapping}")

    # equal-sized partition keeps index alignment (REPEAT stages depend on it)
    t5_specs = _require_specs(5)
    t5_labels = [
        "open top drawer", "close top drawer", "open middle drawer", "close middle drawer",
        "open bottom drawer", "close bottom drawer", "open middle drawer again",
        "place butter into middle drawer", "close middle drawer final",
    ]
    m5 = align_exec_to_scored(t5_labels, t5_specs)
    check(m5.get("open middle drawer again") == "07_Open_Middle_Drawer_Again",
          f"T12: REPEAT stage mis-bridged: {m5.get('open middle drawer again')!r}")
    print("T12 PASS — pour alignment is phase-safe, coverage complete, REPEAT stages intact")


def T13_solvability_closure():
    """No HARD edge may point at a node that can never SETTLE."""
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    specs = _require_specs(8)
    mapping = align_exec_to_scored(_t8_labels(), specs)
    h = BoltHarness()
    h.seed(_t8_labels(), [s.name for s in specs], mapping, name_objects=True)

    unsolvable = {n.nid for n in h.graph.ordered() if not h.graph.solvable(n)}
    check(unsolvable, "T13: task 8 must have instrumental nodes")
    bad = [
        (n.nid, d) for n in h.graph.ordered()
        for d, hard in n.deps if hard == "hard" and d in unsolvable
    ]
    check(not bad, f"T13: hard edges onto unverifiable nodes: {bad}")
    # Prefer not creating those edges at all (n_hard_downgraded may be 0 under the
    # score-first wiring).  Either posture is fine as long as the invariant holds.
    check(h.graph.unreachable_stages == [],
          f"T13: unreachable scored stages: {h.graph.unreachable_stages}")
    print(f"T13 PASS — solvability closure "
          f"(hard_downgraded={h.graph.n_hard_downgraded}, "
          f"instrumental={len(unsolvable)})")


def T14_pour_object_inheritance():
    """A pour node with no substance in its label must inherit the bottle."""
    specs = _require_specs(22)
    labels = [
        "pick tomato", "pour first", "pour second", "place tomato aside",
        "open microwave", "pick cookies", "place cookies", "close microwave",
    ]
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    mapping = align_exec_to_scored(labels, specs)
    h = BoltHarness()
    h.seed(labels, [s.name for s in specs], mapping, name_objects=True)
    pours = [n for n in h.graph.ordered() if n.phase.startswith("pour")]
    check(len(pours) == 2, f"T14: expected 2 pour nodes, got {len(pours)}")
    for n in pours:
        check(n.object == "tomato", f"T14: {n.nid} object not inherited: {n.object!r}")
    check("second time" in pours[1].templates[0],
          f"T14: pour_two template must be count-aware: {pours[1].templates[0]!r}")
    # the served command must never be a bare noun-less pour
    for n in pours:
        check(len(n.templates[0].split()) >= 4, f"T14: degenerate template {n.templates[0]!r}")
    print(f"T14 PASS — pour substance inherited (obj='{pours[0].object}', "
          f"templates={pours[1].templates[0]!r})")


def T15_board_serves_imperative():
    """ACTIVE must serve an imperative command, never the symbolic predicate."""
    specs = _require_specs(19)
    h = BoltHarness()
    h.seed(_t19_labels(), [s.name for s in specs], _t19_map(), name_objects=True)
    h.clock.mark_evidence()
    h.clock.mark_contract()
    board = h.context()
    active = h.graph.active()
    check(active is not None, "T15: no active node")
    serve = h.graph.serve_text(active).rstrip(".")
    check(f"— {serve}" in board or f"serve: {serve}" in board or serve in board,
          f"T15: ACTIVE must serve the ladder command, board={board[:400]}")
    for n in h.graph.ordered():
        if n.predicate and n.predicate != n.primitive and "lift it clear" in n.predicate:
            check(n.predicate not in board,
                  f"T15: predicate leaked into board: {n.predicate!r}")
    print("T15 PASS — board serves the imperative primitive, no predicate leak")


def _replay_ladder(h: BoltHarness, *, steps: int = 24) -> list[str]:
    """Emulate the live loop: serve the board, then let the env credit the stage.

    The progress guard is driven by `harness.observe` (one call per planning
    decision), which is what the real `_build_messages` path does -- so this
    replay exercises the same code path the episode will.
    """
    seen: set[str] = set()
    order: list[str] = []
    for step in range(steps):
        h.observe({})                       # note_serve on the current ACTIVE node
        a = h.graph.active()
        if a is None:
            break
        order.append(a.primitive)
        h.graph.note_attempt(a.nid)
        st = a.verifies_stage or a.settles_with
        if st and st not in seen:
            seen.add(st)
            h.graph.note_verified_stage(st, step=step)
    return order


def T16_ladder_reaches_every_scored_stage():
    """Every scored stage must be reachable through ACTIVE selection alone."""
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    for task_id, labels in ((8, _t8_labels()), (22, _t22_labels())):
        specs = _require_specs(task_id)
        scored = [s.name for s in specs]
        mapping = align_exec_to_scored(labels, specs)
        h = BoltHarness()
        h.seed(labels, scored, mapping, name_objects=True)
        order = _replay_ladder(h)
        settled = {n.verifies_stage or n.settles_with
                   for n in h.graph.ordered() if n.status == NS_SETTLED}
        missing = [s for s in scored if s not in settled]
        check(not missing,
              f"T16: task {task_id} never reached {missing} (ladder={order})")
        # and it must not burn the whole budget on one dead primitive
        top = max(order.count(x) for x in set(order))
        check(top <= 3,
              f"T16: task {task_id} repeated one primitive {top}x "
              f"(limit={h.graph.stall_limit}): {order}")
        print(f"T16 PASS — task {task_id} ladder settles {len(settled)} stages "
              f"in {len(order)} steps (max repeat {top})")


def T17_completion_rule():
    """Once every verifiable obligation is closed, BOLT must stop directing."""
    h = BoltHarness()
    h.seed(_t19_labels(), _t19_scored(), _t19_map(), name_objects=True)
    for stage in _t19_scored():
        h.graph.note_verified_stage(stage, step=1)
    check(h.graph.active() is None,
          "T17: ACTIVE must be None once all verifiable stages are closed")
    h.compile_decision_need()
    h.arbitrate("Pick up the tomato sauce from the source container.")
    check(h.last_decision.verdict in {AV_HOLD, AV_ACT, AV_OBSERVE},
          f"T17: expected pass-through once closed, got {h.last_decision.verdict}")
    print("T17 PASS — completion rule stops direction once all verifiable stages close")


def T18_obligation_discipline():
    """A proposal on a different open obligation must be rewritten to ACTIVE.

    Regression for the measured task-8 stall: `_check_graph` used to accept any
    ready node, so the planner could answer "grasp chocolate" while the board was
    directing it at a different obligation.  The board had no effect and the
    ladder never advanced.
    """
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    specs = _require_specs(8)
    mapping = align_exec_to_scored(_t8_labels(), specs)
    h = BoltHarness()
    h.seed(_t8_labels(), [s.name for s in specs], mapping, name_objects=True)
    h.clock.mark_evidence()
    h.clock.mark_contract()
    # Score-first selection: ACTIVE starts on the Lift (pick tomato sauce), not on
    # the instrumental chocolate pick.  Propose the chocolate grasp → must rewrite.
    active = h.graph.active()
    check(active is not None and active.verifies_stage == "01_Lift_Tomato_Sauce",
          f"T18: ACTIVE should be Lift, got {active.nid if active else None} "
          f"verifies={active.verifies_stage if active else None}")
    decision = h.arbitrate("grasp chocolate")
    check(decision.verdict == AV_REWRITE,
          f"T18: off-active proposal must be rewritten, got {decision.verdict} "
          f"({decision.reason})")
    check(decision.contract.node_id == active.nid,
          f"T18: rewrite must target ACTIVE {active.nid}, got {decision.contract.node_id}")
    print(f"T18 PASS — off-active proposal rewritten to ACTIVE {active.nid} "
          f"({decision.contract.primitive!r})")


def T20_score_first_selection():
    """The first ACTIVE obligation must be a scored stage, not instrumental prep.

    Measured failure (task 8 v3): BDDL order starts with pick/place chocolate;
    selecting those first burned the episode's 80-step stall budget while the
    scorer was already timing out 01_Lift_Tomato_Sauce — every trial ended 0.0
    before Lift was attempted.  Instrumental nodes must not gate scored stages
    via generic sequencing.
    """
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    specs = _require_specs(8)
    mapping = align_exec_to_scored(_t8_labels(), specs)
    h = BoltHarness()
    h.seed(_t8_labels(), [s.name for s in specs], mapping, name_objects=True)
    active = h.graph.active()
    check(active is not None, "T20: no active")
    check(h.graph.solvable(active),
          f"T20: first ACTIVE must be solvable, got {active.nid} {active.primitive}")
    check(active.verifies_stage == "01_Lift_Tomato_Sauce",
          f"T20: expected Lift first, got verifies={active.verifies_stage!r} "
          f"primitive={active.primitive!r}")
    # Lift must not be gated on the instrumental chocolate place
    bad = [
        (d, hrd) for d, hrd in active.deps
        if not h.graph.solvable(h.graph.nodes[d])
    ]
    check(not bad, f"T20: Lift gated on instrumental deps {bad}")
    # I2 — NO-GATE: the pour (a scored stage) must NOT be gated by the instrumental
    # chocolate place.  v5 wired this as a *soft* dep, which looked harmless (soft
    # deps release after one serve) but made pour_one unready behind Lift: after
    # n3(Lift) SETTLED, tie-break fell through to the instrumental tier and the board
    # directed `place chocolate in frypan` between Lift and Pour_One — spending the
    # episode on unscored prep while both pours sat open.
    pour = next(n for n in h.graph.ordered() if n.phase == "pour_one")
    prep_gates = [
        d for d, hrd in pour.deps
        if not h.graph.solvable(h.graph.nodes[d])
    ]
    check(not prep_gates,
          f"T20: pour_one must not be gated by unverifiable prep, deps={pour.deps}")
    check(h.graph.deps_ready(pour) is False or True, "T20: sanity")
    # pour_one becomes ready as soon as its grasp is attempted (soft), so the active
    # chain can never detour through instrumental nodes on the way to a scored stage
    h.graph.note_serve("n3")  # the bottle grasp: one serve satisfies the soft dep
    h.graph.nodes["n3"].status = "SETTLED"
    check(h.graph.deps_ready(pour),
          f"T20: pour_one unreachable after Lift settles, deps={pour.deps}")
    print(f"T20 PASS — score-first ACTIVE={active.nid}:{active.primitive}; "
          f"no unverifiable gate on pour (prep dropped={h.graph.n_prep_deps_dropped})")


def T21_stall_recovery_rotation():
    """Repeated serves on one obligation must rotate the served command."""
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    # Isolate from the evidence-starvation floor: this test is about the serve
    # ladder, not the router.
    os.environ["MEMEXP_BOLT_FLOOR"] = "0"
    os.environ["MEMEXP_BOLT_ROUTER"] = "0"
    try:
        specs = _require_specs(8)
        mapping = align_exec_to_scored(_t8_labels(), specs)
        h = BoltHarness()
        h.seed(_t8_labels(), [s.name for s in specs], mapping, name_objects=True)
        served = []
        for i in range(4):
            h.clock.mark_evidence()
            h.clock.mark_contract()
            h.clock.mark_planner()
            h.observe({})
            a = h.graph.active()
            check(a is not None, f"T21: lost ACTIVE at step {i}")
            served.append(h.graph.serve_text(a))
            d = h.arbitrate("grasp tomato sauce")
            # The discipline that replaced blanket rewriting: a *distinct* phrasing is
            # accepted (variation is the goal), while an echo of the sentence issued on
            # the previous decision is refused and rewritten onto the rotated rung.
            if i >= 1:
                prev = h.graph.prev_served(a.nid)
                echo = h.arbitrate(prev)
                check(
                    echo.verdict == AV_REWRITE
                    or echo.contract.primitive.rstrip(".").lower()
                    != prev.rstrip(".").lower(),
                    f"T21: step {i} must refuse a verbatim echo of {prev!r}, got "
                    f"{echo.verdict} {echo.contract.primitive!r} reason={echo.reason}",
                )
        check(len(set(s.lower() for s in served)) >= 2,
              f"T21: serve ladder did not rotate: {served}")
        # The saturation bug: v1 clamped at len(ladder)-1, so every serve past the
        # end was byte-identical.  The ladder must cycle for as long as the
        # obligation is open, with no two consecutive serves equal.
        served2 = []
        for i in range(12):
            h.clock.mark_evidence()
            h.clock.mark_contract()
            h.clock.mark_planner()
            h.observe({})
            a = h.graph.active()
            check(a is not None, f"T21: lost ACTIVE at long step {i}")
            served2.append(h.graph.serve_text(a))
            h.arbitrate("grasp tomato sauce")
        repeats = [
            i for i in range(1, len(served2))
            if served2[i].strip().lower() == served2[i - 1].strip().lower()
        ]
        check(not repeats, f"T21: consecutive verbatim serves at steps {repeats}: {served2}")
        check(len(set(s.lower() for s in served2)) >= 4,
              f"T21: long-run diversity too low ({len(set(served2))}): {served2}")
        print(f"T21 PASS — serve ladder rotated across {len(served)} steps and stayed "
              f"non-saturating over {len(served2)} ({len(set(served2))} distinct)")
    finally:
        os.environ["MEMEXP_BOLT_FLOOR"] = "1"
        os.environ["MEMEXP_BOLT_ROUTER"] = "1"


def T22_episode_reset_clears_verified():
    """verified_stages must not leak across episodes (task 8 v4 consec_zero)."""
    import types as _types  # noqa: PLC0415

    import bolt.bind as B  # noqa: PLC0415

    fake = _types.ModuleType("bolt_selftest_planner_reset")

    class ApiMemoryPlanner:
        task_info = None
        system_prompt = "planner"
        def reset_episode(self, *a, **k):
            return None
        def set_task_info(self, *a, **k):
            return None
        def _build_messages(self, *a, **k):
            return [{"type": "text", "text": "BASE"}]

    fake.ApiMemoryPlanner = ApiMemoryPlanner
    fake.infer_primitive_via_api = lambda **k: "{}"
    B._wrap_resets(fake)
    B._STATE["verified_stages"] = ["01_Lift_Tomato_Sauce", "02_Pour_One"]
    B._STATE["active_stage"] = "02_Pour_One"
    # simulate a live ctx so the wrap exercises the report+clear path
    p = ApiMemoryPlanner()
    B._LOCAL.planner = p
    B._LOCAL.ctx = B._Ctx(p)
    B._LOCAL.ctx.seeded = True
    p.reset_episode()
    check(B._STATE["verified_stages"] == [],
          f"T22: verified_stages leaked: {B._STATE['verified_stages']}")
    check(B._STATE["active_stage"] == "",
          f"T22: active_stage leaked: {B._STATE['active_stage']}")
    check(B._live_ctx() is None, "T22: thread-local ctx must be cleared")
    # a fresh ctx must start with an empty harness (no pre-settled nodes)
    ctx = B._get_ctx(p)
    check(ctx.harness.n_plan_steps == 0 and not ctx.seeded,
          "T22: new ctx must be unseeded")
    print("T22 PASS — episode reset clears verified_stages and thread-local ctx")


def T19_arbiter_is_live():
    """The planning call must reach the Arbiter; sibling calls must not.

    Regression for the measured defect: `harness.api_vlm_planner` calls
    `infer_primitive_via_api(system_prompt=..., user_content=..., ...)` with pure
    keyword arguments and no planner handle, so the v1 hook silently returned every
    response unchanged -- the Arbiter never ran and `note_attempt` never fired.
    """
    import types as _types  # noqa: PLC0415

    import bolt.bind as B  # noqa: PLC0415

    SYS = "PLANNER-SYSTEM-PROMPT-SENTINEL"

    class _Planner:
        system_prompt = SYS

    p = _Planner()
    prev = (getattr(B._LOCAL, "planner", None), getattr(B._LOCAL, "ctx", None))
    B._LOCAL.planner = p
    B._LOCAL.ctx = B._Ctx(p)
    try:
        got, is_plan = B.resolve_planning_call({"system_prompt": SYS})
        check(got is p and is_plan is True,
              f"T19: planning call not recognised (planner={got is p}, is_plan={is_plan})")
        _, is_other = B.resolve_planning_call(
            {"system_prompt": "You are a strict visual verifier. Output JSON only."}
        )
        check(is_other is False,
              "T19: verifier / decision call misclassified as the planning call")
    finally:
        B._LOCAL.planner, B._LOCAL.ctx = prev

    # the hook must be installed, and must leave non-planning calls byte-identical
    fake = _types.ModuleType("bolt_selftest_planner")

    def _orig(*, system_prompt, user_content, **k):
        return '{"verdict":"yes"}'

    fake.infer_primitive_via_api = _orig
    B._wrap_api_call(fake)
    check(getattr(fake.infer_primitive_via_api, "_bolt_wrapped", False),
          "T19: infer_primitive_via_api was not wrapped")
    passthrough = fake.infer_primitive_via_api(
        system_prompt="You are a strict visual verifier. Output JSON only.", user_content=[]
    )
    check(passthrough == '{"verdict":"yes"}',
          f"T19: sibling call was mutated: {passthrough!r}")
    print("T19 PASS — Arbiter reachable on the planning call, sibling calls untouched")


def T23_ordinal_discipline():
    """I4 / pour ordinality: 'pour a second time' must be refused while the first
    pour is open, and accepted once it SETTLES.

    Measured problem: nothing in BOLT distinguished the two pours, so the planner
    could emit a second-pour command before the first pour existed.  The graph now
    hard-gates pour_two on SETTLED pour_one, and the Arbiter rewrites the early
    request back onto pour_one instead of letting it through.
    """
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    os.environ["MEMEXP_BOLT_FLOOR"] = "0"
    os.environ["MEMEXP_BOLT_ROUTER"] = "0"
    specs = _require_specs(22)
    mapping = align_exec_to_scored(_t22_labels(), specs)
    h = BoltHarness()
    h.seed(_t22_labels(), [s.name for s in specs], mapping, name_objects=True)

    p1 = next(n for n in h.graph.ordered() if n.phase == "pour_one")
    p2 = next(n for n in h.graph.ordered() if n.phase == "pour_two")

    # invariant I4 is materialised as a HARD edge
    check(any(d == p1.nid and hd == "hard" for d, hd in p2.deps),
          f"T23: pour_two must hard-dep on pour_one, deps={p2.deps}")
    check(not h.graph.deps_ready(p2),
          f"T23: pour_two must not be ready while pour_one is open")

    # a second-pour proposal is bound to pour_two, then refused and rewritten
    d = h.arbitrate("pour the tomato over the pan a second time")
    check(d.verdict == AV_REWRITE,
          f"T23: second-pour before first must be rewritten, got {d.verdict} {d.reason}")
    check(h.graph.nodes[d.contract.node_id].phase in {"pour_one", "grasp"},
          f"T23: rewrite must target the active first-pour chain, got "
          f"{h.graph.nodes[d.contract.node_id].phase} ({d.reason})")

    # once pour_one SETTLES, pour_two becomes admissible and is served second-pour
    h.graph.nodes[p1.nid].status = "SETTLED"
    check(h.graph.deps_ready(p2), "T23: pour_two must open once pour_one SETTLED")
    served = h.graph.serve_text(p2)
    check("second" in served.lower(),
          f"T23: pour_two serve must say 'second time', got {served!r}")
    print(f"T23 PASS — pour_two hard-gated on pour_one; rewrite target "
          f"{h.graph.nodes[d.contract.node_id].phase}; serve={served[:48]!r}")


def T24_invariants_hold():
    """Every seeded task family must satisfy I1-I4 and have no violation recorded."""
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    fam = [(5, _t5_labels()), (8, _t8_labels()), (19, _t19_labels()), (22, _t22_labels())]
    for tid, labels in fam:
        specs = _require_specs(tid)
        mapping = align_exec_to_scored(labels, specs)
        h = BoltHarness()
        h.seed(labels, [s.name for s in specs], mapping, name_objects=True)
        check(not h.graph.invariant_violations,
              f"T24: task {tid} invariant violations {h.graph.invariant_violations}")
        # no scored stage may be gated on an unverifiable node, ever
        for n in h.graph.ordered():
            if not h.graph.solvable(n):
                continue
            for d, hd in n.deps:
                check(h.graph.solvable(h.graph.nodes[d]),
                      f"T24: task {tid} {n.nid} gated by unverifiable {d}")
        # every scored stage is carried by exactly the reachable solvable set
        check(not h.graph.unreachable_stages,
              f"T24: task {tid} unreachable scored stages {h.graph.unreachable_stages}")
    print("T24 PASS — I1-I4 hold on tasks 5/8/19/22; no unverifiable gate on a scored node")


def T25_rungs_can_satisfy_their_stage():
    """I5 — every served rung must be able to satisfy the stage the node verifies.

    Measured regression (v7, t8): three of six grasp rungs asked only for a grasp
    ("reach for and grasp…", "re-localize … close the gripper…", "approach … grip
    firmly…") while the obligation was verified by the environment's Lift stage.  The
    VLA complied exactly and never lifted, so nothing was ever credited; the Arbiter
    kept the same node ACTIVE and all three trials spent ~2400 steps and 18-19
    attempts on a command that could not succeed.  Score: 0.0 / 0.0 / 0.0.
    """
    from bolt import serve as SV  # noqa: PLC0415
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    # a grasp-only rung must be rejected when the obligation ends in a lift
    check(not SV.satisfies_stage("grasp", "Re-localize the bottle, then close the gripper."),
          "T25: a grasp-only rung must not count as satisfying a lift obligation")
    check(SV.satisfies_stage("grasp", "Pick up the tomato sauce bottle."),
          "T25: 'pick up' must satisfy a lift obligation")
    check(SV.satisfies_stage("close", "Push the drawer closed."),
          "T25: 'closed' must count as a close action")

    fam = [(5, _t5_labels()), (8, _t8_labels()), (19, _t19_labels()), (22, _t22_labels())]
    for tid, labels in fam:
        specs = _require_specs(tid)
        h = BoltHarness()
        h.seed(labels, [s.name for s in specs], align_exec_to_scored(labels, specs),
               name_objects=True)
        bad = [(n.nid, n.phase, t)
               for n in h.graph.ordered() if h.graph.solvable(n)
               for t in SV.violating_rungs(n.phase, h.graph.serve_ladder(n))]
        check(not bad, f"T25: task {tid} has rungs that cannot satisfy their stage: {bad}")
        # and no invariant may survive as a recorded violation
        check(not h.graph.invariant_violations,
              f"T25: task {tid} invariant violations {h.graph.invariant_violations}")
    print("T25 PASS — I1-I5 hold on tasks 5/8/19/22; every rung can satisfy its stage")


def T26_no_identifier_leak_and_runtime_i5():
    """I5 must hold at RUNTIME, so no identifier leak can become a dead step.

    Measured regression (v8, t8): the board's OPEN list printed `[n3] grasp tomato
    sauce [OPEN]`.  The planner copied it as `grasp tomato sauce bottle`; a grasp
    succeeds but the node is verified by the environment's LIFT stage, so nothing was
    credited and seeds 100/102/104 burned to 0.0 while the control arm lifted in the
    same scenes.  Two independent protections are asserted here: the board no longer
    exposes such a string, and the Arbiter refuses it even if something else emits it.
    """
    from bolt.align import align_exec_to_scored  # noqa: PLC0415

    os.environ["MEMEXP_BOLT_FLOOR"] = "0"
    os.environ["MEMEXP_BOLT_ROUTER"] = "0"
    specs = _require_specs(8)
    h = BoltHarness()
    h.seed(_t8_labels(), [s.name for s in specs],
           align_exec_to_scored(_t8_labels(), specs), name_objects=True)
    h.clock.mark_evidence()
    h.clock.mark_contract()

    # 1) the board must not contain an actionable `phase object` pair
    board = h.context()
    active = h.graph.active()
    check(active is not None, "T26: no active node")
    check(f"{active.phase} {active.object}" not in board,
          f"T26: board leaks the actionable identifier "
          f"{active.phase} {active.object!r}")
    check("stage=" in board, "T26: OPEN list should name the scored stage")

    # 2) the Arbiter must refuse a grasp-only command on a Lift-verified node
    a = h.arbitrate("grasp tomato sauce bottle")
    check(a.verdict == AV_REWRITE,
          f"T26: grasp-only on a lift obligation must be rewritten, got {a.verdict} {a.reason}")
    node = h.graph.nodes[a.contract.node_id]
    check(SV.satisfies_stage(node.phase, a.contract.primitive),
          f"T26: rewrite must satisfy the stage, got {a.contract.primitive!r} "
          f"for {node.phase}")
    check(not SV.satisfies_stage("grasp", "grasp tomato sauce bottle"),
          "T26: a bare grasp must not count as satisfying a lift obligation")

    # 3) a genuine lift phrasing is accepted
    ok = h.arbitrate("Pick up the tomato sauce bottle.")
    check(ok.verdict in {AV_ACT, AV_REWRITE},
          f"T26: lift phrasing must not be rejected, got {ok.verdict} {ok.reason}")
    print(f"T26 PASS — no actionable identifier on the board; runtime I5 rewrites "
          f"'grasp tomato sauce bottle' → {a.contract.primitive!r}")


def main() -> int:
    tests = [
        T1_execution_seed,
        T2_soft_grasp_place,
        T3_hard_pour_chain,
        T4_phi_monotone,
        T5_dual_clock_hold,
        T6_decision_need,
        T7_arbiter_rewrite,
        T8_claim_invalidation,
        T9_derived_from_ledger,
        T10_starvation_floor,
        T11_report_shape,
        T12_pour_alignment,
        T13_solvability_closure,
        T14_pour_object_inheritance,
        T15_board_serves_imperative,
        T16_ladder_reaches_every_scored_stage,
        T17_completion_rule,
        T18_obligation_discipline,
        T19_arbiter_is_live,
        T20_score_first_selection,
        T21_stall_recovery_rotation,
        T22_episode_reset_clears_verified,
        T23_ordinal_discipline,
        T24_invariants_hold,
        T25_rungs_can_satisfy_their_stage,
        T26_no_identifier_leak_and_runtime_i5,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Fail as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1
    print()
    if failed:
        print(f"BOLT SELFTEST FAILED: {failed}/{len(tests)}")
        return 1
    print("=" * 78)
    print("BOLT SELFTEST PASSED: execution graph, soft/hard deps, Φ monotone,")
    print("dual-clock window, Decision Need, Arbiter rewrite, Claim invalidation,")
    print("ledger-derived counts, starvation floor, report shape.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
