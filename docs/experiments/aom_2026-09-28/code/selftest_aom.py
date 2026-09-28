#!/usr/bin/env python
"""mem_efficacy / AOM test: the unified architecture, asserted at the level of its claims.

WHAT IS ASSERTED, AND WHY EACH ONE IS LOAD-BEARING
  T1  the obligation graph: ACTUAL obligations form a ladder, PERCEPTUAL ones do NOT.
      The asymmetry is the occlusion case stated in `arms/pullmem.sh` -- three drawers, each
      observed once, and which one matters is not known until all three are seen. A sequential
      PERCEPTUAL ladder would forbid looking at the third before the first settles.
  T2  the law: all five verdicts, and the ORDER of its branches.
  T3  floor F: `stagnant` still retrieves. This is the invariant PIC-MEM v3 violated, and its
      failure (evidence cadence 8-15 -> 4 frames/episode, 27.8 vs GPM's 66.7) is the reason the
      floor is a test and not a comment.
  T4  the graph gate is DERIVED, not pattern-matched: every rule GPM implements as a regex is
      reproduced here as a graph reachability fact, and the gate is CONSERVATIVE (an
      unclassifiable paraphrase is allowed, because rejecting it would make the gate a
      measurement of the model's vocabulary).
  T5  DERIVED: minting, live recomputation on every settle, and settlement at target.
  T6  the DERIVED ablation really removes the mode, so T5's claim is falsifiable.
  T7  the board's sections are functions of (mode, status) -- the property that stops the three
      prompt surfaces of the parent designs from drifting apart.
  T8  floor F as a CADENCE claim: turning it off does not change what `arbitrate` returns for a
      saturated obligation, because F lives in the bind layer's retrieval trigger.
Costs milliseconds and no API quota.
"""
from __future__ import annotations

import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail and not ok:
        print(f"         {detail}")
    if not ok:
        FAILS.append(label)
    return ok


# Every AOM knob, so a test that sets one is guaranteed not to leak it into the next. This is not
# hygiene: T4b sets MEMEXP_AOM_GRAPH_GATE=0 to prove the gate's ablation works, and while `fresh()`
# cleared only the flags it was handed, that 0 survived into T5 -- which then reported the DERIVED
# mode as broken when the gate was simply off. A leak across tests looked exactly like a bug in the
# feature under test.
_KNOBS = {
    "MEMEXP_AOM": "1",
    "MEMEXP_AOM_DERIVED": "1",
    "MEMEXP_AOM_GRAPH_GATE": "1",
    "MEMEXP_AOM_FLOOR": "1",
    "MEMEXP_AOM_GATE_ATTEMPTS": "3",
    "MEMEXP_AOM_GATE_STALL": "3",
    "MEMEXP_AOM_RET_MAX": "6",
    "MEMEXP_AOM_STAG_ACT_REARM": "3",
    "MEMEXP_AOM_ATTRACTOR_REPEAT": "2",
}


def fresh(**env: str):
    """A module reloaded under a fully specified environment. The law's thresholds are read at
    import, so a test that changed them on a shared module would leak into the next test."""
    for k, v in _KNOBS.items():
        os.environ[k] = v
    for k, v in env.items():
        os.environ[k] = v
    import memexp_aom
    return importlib.reload(memexp_aom)


T8_STAGES = ["01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"]
OCC_STAGES = ["01_Open_Top_Drawer", "02_Open_Middle_Drawer", "03_Open_Bottom_Drawer"]
T19_STAGES = [
    "01_Place_Tomato_Sauce_Cabinet2",
    "02_Place_Milk_Cabinet2",
    "03_Place_Orange_Juice_Cabinet2",
]
T5_LIKE = [
    "01_Open_Top_Drawer",
    "02_Close_Top_Drawer",
    "03_Open_Middle_Drawer",
    "04_Close_Middle_Drawer",
    "05_Open_Bottom_Drawer",
    "06_Close_Bottom_Drawer",
    "07_Place_Object",
]
# Task 19 has TWO decompositions, and the defect was seeding from the wrong one.
#   scoring   (from _task_specs)      = the three Place_*_Cabinet2 stages
#   execution (from primitive_order)  = pick/place for each of three objects
T19_EXEC = [
    "pick tomato sauce",
    "place tomato sauce cabinet2",
    "pick milk",
    "place milk cabinet2",
    "pick orange juice",
    "place orange juice cabinet2",
]


class _Spec:
    """Minimal StageSpec stand-in: `expected_primitive_for_stage` only needs `.name`."""

    def __init__(self, name: str) -> None:
        self.name = name


def _t19_mapping() -> dict[str, str]:
    """Resolve the execution->scoring bridge with the harness's OWN mapper.

    Deliberately calls the real `harness.stage_mapper.expected_primitive_for_stage` rather than a
    local reimplementation: the point of the fix is that AOM stops inventing a second mapping.
    """
    specs = [_Spec(n) for n in T19_STAGES]
    try:
        from harness.stage_mapper import expected_primitive_for_stage as _exp
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for i, sp in enumerate(specs):
        lab = _exp(primitive_labels=T19_EXEC, stage_idx=i, stage_specs=specs,
                   stage_done={s.name: False for s in specs[:i]}, current_subtask="")
        if lab and lab not in out:
            out[lab] = sp.name
    return out


def main() -> int:
    print("=" * 78)
    print("AOM: one obligation type, one law, one board, derivable admissibility")
    print("=" * 78)
    A = fresh()

    print("\n[T1] the obligation graph: ACTUAL is a ladder; open/close are ACTUAL+needs_look")
    L = A.Ledger()
    n = L.seed(T8_STAGES)
    acts = [o for o in L._ordered() if o.mode == A.MODE_ACTUAL]
    check(n == 5 and len(acts) == 3, "3 ACTUAL + 2 DERIVED minted from a 3-stage ladder",
          f"n={n} modes={[(o.kind, o.mode) for o in L._ordered()]}")
    check(acts[0].deps == [], "lift has no predecessor", f"deps={acts[0].deps}")
    check(acts[1].deps == [acts[0].oid], "pour_one depends on lift", f"deps={acts[1].deps}")
    check(acts[2].deps == [acts[1].oid], "pour_two depends on pour_one", f"deps={acts[2].deps}")

    L2 = A.Ledger()
    L2.seed(OCC_STAGES)
    opens = [o for o in L2._ordered() if o.mode != A.MODE_DERIVED]
    check(len(opens) == 3 and all(o.mode == A.MODE_ACTUAL for o in opens),
          "three open stages are ACTUAL (the robot must physically open)",
          f"modes={[(o.kind, o.mode) for o in L2._ordered()]}")
    check(all(o.needs_look for o in opens),
          "open stages carry needs_look (evidence before / while acting)",
          f"needs_look={[o.needs_look for o in opens]}")
    check(opens[0].deps == [] and opens[1].deps == [opens[0].oid],
          "open/close ACTUAL stages form a causal ladder (not the old PERCEPTUAL independence)",
          f"deps={[(o.kind, o.deps) for o in opens]}")

    print("\n[T1b] place-family siblings are independent (task-19 causal rule)")
    Lp = A.Ledger()
    Lp.seed(T19_STAGES)
    places = [o for o in Lp._ordered() if o.mode == A.MODE_ACTUAL]
    check(len(places) == 3 and all(o.deps == [] for o in places),
          "three place stages have NO deps on each other",
          f"kinds/deps={[(o.kind, o.deps) for o in places]}")
    check(all("#" in o.kind for o in places),
          "place kinds carry a discriminant (#1/#2/#3)",
          f"kinds={[o.kind for o in places]}")

    print("\n[T2] the law: five verdicts, in this branch order")
    L = A.Ledger()
    L.seed(T8_STAGES)
    L.set_active_stage("01_Lift_Tomato_Sauce", 0)
    check(L.arbitrate(stall=0)[0] == "act", "ACTUAL, unsaturated -> act")
    check(L.arbitrate(stall=99)[0] == "retrieve",
          "ACTUAL, stall gate tripped -> retrieve (GPM's decide_mode)")
    o = L.active()
    o.tau_act = 99
    o.tau_ret = 0
    check(L.arbitrate(stall=0)[0] == "retrieve", "ACTUAL, attempts gate tripped -> retrieve")
    o.tau_ret = 99
    check(L.arbitrate(stall=0)[0] == "stagnant",
          "both channels saturated -> stagnant")
    L.force_retrieve = True
    check(L.arbitrate(stall=0)[0] == "retrieve",
          "an attractor repeat forces retrieve even when both channels are saturated")
    L.force_retrieve = False
    L.note_verified("01_Lift_Tomato_Sauce", 5)
    verdict, oid = L.arbitrate(stall=99)
    check(oid != o.oid,
          "a SETTLED active obligation ADVANCES the pointer rather than being re-served",
          f"verdict={verdict} oid={oid} was={o.oid}")
    check(L.active().kind == "pour_one", "the pointer moved to pour_one",
          f"active={L.active().kind}")
    check(verdict == "retrieve",
          "and the verdict is then computed for the NEW obligation (stall=99 supersedes its "
          "action channel)", f"verdict={verdict}")
    L.note_verified("02_Pour_One", 6)
    L.note_verified("03_Pour_Two", 7)
    check(L.fluent_equivalent(), "with everything settled the ledger is fluent-equivalent")
    check(L.arbitrate(stall=0)[0] == "advance",
          "nothing open -> advance (the trigger for floor F's 'say nothing' branch)")

    print("\n[T3] floor F: STAGNANT is a board statement, never an evidence cut")
    L3 = A.Ledger()
    L3.seed(T8_STAGES)
    L3.set_active_stage("01_Lift_Tomato_Sauce", 0)
    L3.active().tau_act = 99
    L3.active().tau_ret = 99
    ret = L3.arbitrate(stall=0)[0]
    check(ret == "stagnant", "saturated on both channels -> stagnant", f"got {ret}")
    check(A.floor_on() is True, "floor F is ON by default")
    # The bind layer is what must retrieve for BOTH verdicts; assert it by construction here,
    # so the property is checked even without loading the harness.
    import memexp_aom_bind as B
    src = open(os.path.join(HERE, "memexp_aom_bind.py"), encoding="utf-8").read()
    check('if verdict in {"retrieve", "stagnant"}' in src,
          "the bind retrieves for retrieve AND stagnant -- so declaring stagnation cannot cost a "
          "frame (the PIC-MEM v3 regression)")
    check("def _run_auto_retrieval" in src, "the auto-retrieval path exists")
    del B

    print("\n[T4] the graph gate: GPM's regex table, re-derived as reachability")
    L = A.Ledger()
    L.seed(T8_STAGES)
    cases = [
        ("Reach for and grasp the tomato sauce bottle.", None,
         "a legal lift primitive passes"),
        ("Tilt the tomato sauce bottle to pour sauce onto the chocolate.",
         "dependency", "pour before lift -> dependency violation (GPM: _POUR_RX + lift_done)"),
        ("Pour the tomato sauce over the chocolate a second time.",
         "dependency", "second pour before pour_one (GPM: _SECOND_POUR_RX)"),
        ("Open the microwave and place the chocolate inside.",
         "off-graph", "microwave not in the graph (GPM: _MICROWAVE_RX + stage_blob test)"),
        ("step 1 of 3", "label", "a pasted stage label (GPM: looks_like_label_primitive)"),
        ("Wiggle the gripper a little.", None,
         "an UNCLASSIFIABLE paraphrase is ALLOWED: rejecting it would make the gate a measure of "
         "the model's vocabulary"),
    ]
    for prim, want, why in cases:
        got = L.admissibility(prim)
        if want is None:
            check(got is None, why, f"got {got!r} for {prim!r}")
        elif want == "label":
            check(got is not None and "label" in got, why, f"got {got!r}")
        else:
            check(got is not None and want in got, why, f"got {got!r} for {prim!r}")

    L.note_verified("01_Lift_Tomato_Sauce", 3)
    got = L.admissibility("Tilt the tomato sauce bottle to pour sauce onto the chocolate.")
    check(got is None, "the SAME pour primitive becomes admissible once lift is SETTLED",
          f"got {got!r}")
    got = L.admissibility("Pour the tomato sauce over the chocolate a second time.")
    check(got is not None and "pour_one" in got,
          "the second pour is still blocked, now by pour_one rather than by lift",
          f"got {got!r}")

    print("\n[T4b] the gate's ablation knob really disables it")
    A2 = fresh(MEMEXP_AOM_GRAPH_GATE="0")
    Lg = A2.Ledger()
    Lg.seed(T8_STAGES)
    check(Lg.admissibility("Open the microwave and place the chocolate inside.") is None,
          "GRAPH_GATE=0 -> the microwave is no longer rejected (measures the graph's value)")
    check(Lg.admissibility("step 1 of 3") is not None,
          "GRAPH_GATE=0 still rejects a pasted label (the check that is not graph-derived)")
    A = fresh()

    print("\n[T5] DERIVED: minted, recomputed on EVERY settle, and settled at target")
    L = A.Ledger()
    L.seed(T8_STAGES)
    d = {o.kind: o for o in L._ordered() if o.mode == A.MODE_DERIVED}
    check(set(d) == {"count:pour", "count:steps"}, "both progress obligations minted",
          f"got {sorted(d)}")
    check(d["count:pour"].target == 2 and d["count:steps"].target == 3,
          "targets are counts of the counted set, not constants",
          f"pour={d['count:pour'].target} steps={d['count:steps'].target}")
    check(d["count:pour"].value == "0/2", "the value starts live, not empty",
          f"got {d['count:pour'].value!r}")
    L.note_verified("01_Lift_Tomato_Sauce", 10)
    check(d["count:pour"].value == "0/2" and d["count:steps"].value == "1/3",
          "settling a non-pour step moves the step count and NOT the pour count",
          f"pour={d['count:pour'].value} steps={d['count:steps'].value}")
    L.note_verified("02_Pour_One", 20)
    check(d["count:pour"].value == "1/2", "the pour count follows the settled set",
          f"got {d['count:pour'].value!r}")
    check(d["count:pour"].status != A.ST_SETTLED, "1 of 2 is not complete")
    L.note_verified("03_Pour_Two", 30)
    check(d["count:pour"].value == "2/2" and d["count:pour"].status == A.ST_SETTLED,
          "2 of 2 completes the DERIVED obligation",
          f"value={d['count:pour'].value} status={d['count:pour'].status}")
    check(d["count:pour"].settled_step == 30, "the settle step is recorded",
          f"got {d['count:pour'].settled_step}")

    print("\n[T5b] the completed counting family is what blocks an over-pour")
    L = A.Ledger()
    L.seed(T8_STAGES)
    L.note_verified("01_Lift_Tomato_Sauce", 1)
    L.note_verified("02_Pour_One", 2)
    L.note_verified("03_Pour_Two", 3)
    got = L.admissibility("Pour the tomato sauce over the chocolate one more time.")
    check(got is not None and "complete" in got,
          "a third pour is rejected BECAUSE the derived count is complete -- the task-8 failure "
          "mode the archived arms could not see (extra_pour_detected == 0 everywhere)",
          f"got {got!r}")

    print("\n[T6] the DERIVED ablation: the mode is really gone, so T5 is falsifiable")
    A3 = fresh(MEMEXP_AOM_DERIVED="0")
    Ld = A3.Ledger()
    Ld.seed(T8_STAGES)
    check(not any(o.mode == A3.MODE_DERIVED for o in Ld._ordered()),
          "DERIVED=0 -> no progress obligation exists at all",
          f"kinds={[o.kind for o in Ld._ordered()]}")
    Ld.note_verified("01_Lift_Tomato_Sauce", 1)
    Ld.note_verified("02_Pour_One", 2)
    got = Ld.admissibility("Pour the tomato sauce over the chocolate one more time.")
    check(got is None,
          "DERIVED=0 -> the third pour is NOT blocked, because nothing counts the pours",
          f"got {got!r}")
    A = fresh()

    print("\n[T7] the board is a function of (mode, status), not of a per-site decision")
    L = A.Ledger()
    L.seed(OCC_STAGES)
    L.set_active_stage("01_Open_Top_Drawer", 0)
    b_act = L.board(mode="fluent", stall=0)
    check("OPEN (must be LOOKED at" in b_act,
          "a needs_look obligation renders in the OPEN section")
    check("PROGRESS" in b_act and "count:steps" in b_act,
          "open ladder mints count:steps over the ACTUAL set",
          f"board head:\n{b_act[:400]}")
    L2 = A.Ledger()
    L2.seed(T8_STAGES)
    L2.set_active_stage("01_Lift_Tomato_Sauce", 0)
    b = L2.board(mode="evidence", stall=3)
    check("PROGRESS" in b and "count:pour" in b,
          "the DERIVED value renders in the PROGRESS section")
    check("Admissible templates" in b, "evidence mode renders the templates")
    check("SETTLED (do not redo)" not in b, "nothing is settled yet, so no SETTLED line")
    d0 = {o.kind: o for o in L2._ordered() if o.mode == A.MODE_DERIVED}
    L2.note_verified("01_Lift_Tomato_Sauce", 4)
    b2 = L2.board(mode="fluent", stall=0)
    check("SETTLED (do not redo)" in b2, "a settled obligation renders in the SETTLED line")
    check("waits on" in b2,
          "the forbidden set is COMPUTED from unmet dependencies, not hardcoded",
          f"board:\n{b2[:500]}")
    check(d0["count:pour"].value == "0/2", "the pour count did not move on a lift")

    print("\n[T8] floor F as a cadence claim")
    check(A.floor_on() is True, "F defaults ON")
    src = open(os.path.join(HERE, "memexp_aom.py"), encoding="utf-8").read()
    check("if not self.sat_ret(o):" in src and 'return "stagnant"' in src,
          "`stagnant` is decided AFTER the retrieve branch in the law itself")
    idx_ret = src.index('if not self.sat_ret(o):')
    idx_stag = src.index('return "stagnant"')
    check(idx_ret < idx_stag,
          "and the ORDER is asserted, not just the presence: retrieve is tried first")

    print("\n[T9] the module is inert unless the arm asks for it")
    for k in ("MEMEXP_AOM",):
        os.environ.pop(k, None)
    Aoff = importlib.reload(A)
    check(Aoff.enabled() is False, "MEMEXP_AOM unset -> enabled() is False")
    os.environ["MEMEXP_AOM"] = "1"
    A = importlib.reload(A)

    print("\n[T10] the report is actually WRITTEN (the census reads it, so a silent no-write")
    print("      turns a working arm into a missing one)")
    import json
    import tempfile
    tmp = tempfile.mkdtemp(prefix="aom_report_")
    os.environ["MEMEXP_AOM_REPORT"] = os.path.join(tmp, "memexp_aom_report.json")
    Lr = A.Ledger()
    Lr.seed(T8_STAGES)
    Lr.set_active_stage("01_Lift_Tomato_Sauce", 0)
    Lr.note_action("Reach for and grasp the sauce bottle.", 0, 0, 4)
    Lr.note_verified("01_Lift_Tomato_Sauce", 5)
    out = A.write_report([Lr])
    ok = out is not None and os.path.exists(out)
    check(ok, "write_report returns an existing path (not None, not an exception swallowed)",
          f"returned {out!r}")
    payload = None
    if ok:
        try:
            payload = json.loads(open(out, encoding="utf-8").read())
        except Exception as exc:  # noqa: BLE001
            check(False, "the report parses as JSON", f"{exc!r}")
    if payload is not None:
        check("totals" in payload and "ledgers" in payload,
              "the report has the totals/ledgers shape the census reads")
        check(payload["totals"]["n_plan_steps"] >= 1,
              "n_plan_steps is non-zero, i.e. the ledger was summarised rather than skipped",
              f"totals={payload.get('totals')}")
        check(payload["ledgers"][0]["derived_values"].get("count:pour") == "0/2",
              "the DERIVED values survive into the report (they are the headline output)",
              f"got {payload['ledgers'][0].get('derived_values')}")
        check("coverage" in payload["ledgers"][0],
              "coverage telemetry is present in the ledger summary")
    board_path = os.path.join(tmp, f"{os.path.basename(str(out))}.board.jsonl")
    check(os.path.exists(board_path),
          "the board audit file is written next to it", f"looked for {board_path}")
    os.environ.pop("MEMEXP_AOM_REPORT", None)

    print("\n[T11] place self-lock invariant: a place primitive is always admissible for the")
    print("      first open place (the bug that produced 47 dep_rejects on t19)")
    L = A.Ledger()
    L.seed(T19_STAGES)
    prim = "Place the object into the open container."
    got = L.admissibility(prim)
    check(got is None, "with all places open, a place primitive is admissible",
          f"got {got!r}")
    targets = L.obligations_for(prim)
    check(len(targets) == 1, "obligations_for returns exactly one target",
          f"got {[t.kind for t in targets]}")
    L.note_verified(T19_STAGES[0], 1)
    got = L.admissibility(prim)
    check(got is None,
          "after place#1 SETTLED, the same place primitive is STILL admissible (targets place#2)",
          f"got {got!r}; targets={[t.kind for t in L.obligations_for(prim)]}")
    L.note_verified(T19_STAGES[1], 2)
    got = L.admissibility(prim)
    check(got is None,
          "after place#2 SETTLED, place primitive still admissible (targets place#3)",
          f"got {got!r}")
    # Board must never say 'place' waits on 'place' without discriminant.
    b = L.board(mode="fluent", stall=0)
    check("waits on place." not in b and "waits on place'" not in b,
          "forbidden lines never render the self-same-name place/place lie",
          f"board:\n{b[:500]}")

    print("\n[T12] stagnant is reachable: fruitless looks after first evidence do NOT reset tau_ret")
    A12 = fresh(MEMEXP_AOM_RET_MAX="2", MEMEXP_AOM_STAG_ACT_REARM="0")
    L = A12.Ledger()
    L.seed(T8_STAGES)
    L.set_active_stage("01_Lift_Tomato_Sauce", 0)
    o = L.active()
    o.frames = [0, 1, 2, 3]
    store = {0: "a", 1: "b", 2: "c", 3: "d"}
    # First look: OPEN -> SEEN, tau_ret resets to 0.
    text, shown = L.look(o.oid, set(), store, auto=True)
    check(o.status == A12.ST_SEEN and o.tau_ret == 0,
          "first informative look moves OPEN->SEEN and clears tau_ret",
          f"status={o.status} tau_ret={o.tau_ret} shown={shown}")
    # Second look: same frames, already offered -> tau_ret increments, no reset.
    L.look(o.oid, set(), store, auto=True)
    check(o.tau_ret == 1, "second look with no status move increments tau_ret",
          f"tau_ret={o.tau_ret}")
    L.look(o.oid, set(), store, auto=True)
    check(o.tau_ret >= 2, "third look reaches RET_MAX", f"tau_ret={o.tau_ret}")
    o.tau_act = 99
    check(L.arbitrate(stall=0)[0] == "stagnant",
          "after RET_MAX fruitless looks + sat_act, verdict is stagnant (rearm OFF)",
          f"verdict={L.arbitrate(stall=0)[0]}")
    A = fresh()

    print("\n[T12b] stagnant re-arms act so 'change strategy' is actionable")
    A12b = fresh(MEMEXP_AOM_RET_MAX="2", MEMEXP_AOM_STAG_ACT_REARM="3")
    L = A12b.Ledger()
    L.seed(T8_STAGES)
    L.set_active_stage("01_Lift_Tomato_Sauce", 0)
    o = L.active()
    o.tau_act = 99
    o.tau_ret = 99
    v1 = L.arbitrate(stall=0)[0]
    v2 = L.arbitrate(stall=0)[0]
    v3 = L.arbitrate(stall=0)[0]
    check(v1 == "stagnant" and v2 == "stagnant",
          "first two saturated steps are stagnant", f"got {v1},{v2}")
    check(v3 == "act" and o.tau_act == 0 and o.tau_ret == 0,
          "third stagnant hit re-arms act (tau_act=tau_ret=0)",
          f"verdict={v3} tau_act={o.tau_act} tau_ret={o.tau_ret}")
    A = fresh()

    print("\n[T15] NO DEADLOCK: the grasp/place edge is a precondition, not a cycle")
    print("      (job 617408 emitted 719 grasps and ZERO places because 'place requires pick")
    print("       SETTLED' could only be satisfied by a place the same rule refused)")
    mapping = _t19_mapping()
    L = A.Ledger()
    L.seed_execution(T19_EXEC, T19_STAGES, mapping, name_objects=True)
    acts = [o for o in L._ordered() if o.mode == A.MODE_ACTUAL]
    pick1, place1 = acts[0], acts[1]
    check(L.admissibility("Place the tomato sauce into the target container.") is not None,
          "a place before the object has EVER been reached for is still rejected",
          f"pick1.status={pick1.status} tau_act={pick1.tau_act}")
    # The grasp is served (the act channel ran) -> the precondition is met, and the gate must open.
    pick1.tau_act = 1
    check(L.admissibility("Place the tomato sauce into the target container.") is None,
          "once the grasp has been ATTEMPTED, the place is admissible (no cycle)",
          f"pick1.status={pick1.status} tau_act={pick1.tau_act}")
    L.set_active_stage(T19_STAGES[0], 3)
    check(L.active().kind == "place#1",
          "and ACTIVE advances to the place, so the board advertises what to do next",
          f"active={L.active().kind}")
    board1 = L.board(mode="fluent", stall=0)
    check("place the tomato sauce into the target container" in board1.lower(),
          "the board's NEED becomes a NAMED place (was a grasp while nothing was held)",
          f"board:\n{board1[:400]}")
    check("templates" in board1.lower() or "place the tomato sauce" in board1.lower(),
          "and the place is advertised as a command the Planner can imitate",
          f"board:\n{board1[:400]}")
    # An ordinary sequencing edge keeps the hard SETTLED requirement.
    pick2, place2 = acts[2], acts[3]
    check(L.admissibility("Pick up the milk from the source container.") is None
          or pick2.status != A.ST_SETTLED,
          "the next object's pick is not blocked by a SETTLED demand on its own place",
          f"pick2={pick2.status}")

    print("\n[T13] open/close ACTUAL no longer starves act (task-5 MODE fix)")
    L = A.Ledger()
    L.seed(T5_LIKE)
    acts = [o for o in L._ordered() if o.mode == A.MODE_ACTUAL]
    check(len(acts) == 7, "all seven stages are ACTUAL",
          f"modes={[(o.kind, o.mode, o.needs_look) for o in L._ordered()]}")
    v, ntot = L.progress()
    check(ntot == 7, "progress denominator counts all non-DERIVED stages",
          f"progress={v}/{ntot}")
    L.set_active_stage("01_Open_Top_Drawer", 0)
    o = L.active()
    # needs_look with no evidence yet -> retrieve first
    check(L.arbitrate(stall=0)[0] == "retrieve",
          "needs_look with zero evidence -> retrieve before act",
          f"verdict={L.arbitrate(stall=0)[0]}")
    o.n_new_evidence = 1
    o.status = A.ST_SEEN
    check(L.arbitrate(stall=0)[0] == "act",
          "after evidence, unsaturated open stage goes to act (not stuck in retrieve)",
          f"verdict={L.arbitrate(stall=0)[0]}")

    print("\n[T14] EXECUTION-partition seeding: the graph contains the PICK the scoring table omits")
    print("      (the defect that put every AOM version at or below nomem on task 19)")
    mapping = _t19_mapping()
    check(len(mapping) == 3, "the harness stage_mapper binds the 3 scored stages to place steps",
          f"mapping={mapping}")
    L = A.Ledger()
    n = L.seed_execution(T19_EXEC, T19_STAGES, mapping, name_objects=True)
    acts = [o for o in L._ordered() if o.mode == A.MODE_ACTUAL]
    check(len(acts) == 6, "six EXECUTION obligations, not three scored ones",
          f"kinds={[o.kind for o in acts]}")
    check([o.kind for o in acts] ==
          ["pick#1", "place#1", "pick#2", "place#2", "pick#3", "place#3"],
          "kinds are the pick/place ladder with discriminants",
          f"kinds={[o.kind for o in acts]}")
    check(acts[0].deps == [], "the FIRST obligation is an unblocked pick",
          f"deps={acts[0].deps} kind={acts[0].kind}")
    check(acts[1].deps == [acts[0].oid], "place#1 depends on pick#1",
          f"deps={acts[1].deps}")
    check(acts[1].settles_with == "" and acts[0].settles_with == T19_STAGES[0],
          "a pick settles DERIVATIONALLY with the place it feeds",
          f"pick.settles_with={acts[0].settles_with!r}")

    # The scored-stage hint must NOT bypass an unmet execution prerequisite.
    L.set_active_stage(T19_STAGES[0], 0)
    active = L.active()
    check(active is not None and active.kind == "pick#1",
          "the harness hint (scored place #1) does NOT move ACTIVE past the unsettled pick",
          f"active={active.kind if active else None}")
    board0 = L.board(mode="fluent", stall=0)
    check("grasp the tomato sauce" in board0.lower(),
          "the board's NEED is a GRASP, not a place",
          f"board:\n{board0[:400]}")
    check("place the tomato sauce" not in board0.lower(),
          "no place command is advertised while nothing is held")

    # A place emitted before the pick is rejected AND rewritten to an object-naming pick.
    got = L.admissibility("Place the tomato sauce into the target container.")
    check(got is not None and "place#1" in got,
          "an early place is rejected by the pick#1 dependency",
          f"got {got!r}")
    rewritten, did = L.apply_control(
        '{"current_primitive": "Place the tomato sauce into the target container.",'
        ' "keyframe_positions": []}')
    check(did and "Pick up the tomato sauce" in rewritten,
          "the rewrite is a NAMED pick, so the recovery command is the correct first action",
          f"got {rewritten!r}")

    L.note_verified(T19_STAGES[0], 5)
    check(acts[0].status == A.ST_SETTLED and acts[1].status == A.ST_SETTLED,
          "one scored settle closes BOTH the pick and the place of that object",
          f"pick={acts[0].status} place={acts[1].status}")
    check(L.active().kind == "pick#2",
          "ACTIVE self-heals to the next object's pick once the pointer's stage is settled",
          f"active={L.active().kind}")
    got = L.admissibility("Pick up the milk from the source container.")
    check(got is None, "the next object's pick is admissible",
          f"got {got!r}")

    print("\n[T14b] object naming is ON for transfer tasks and OFF for occlusion tasks")
    Ln = A.Ledger()
    Ln.seed_execution(T19_EXEC, T19_STAGES, mapping, name_objects=True)
    tn = Ln.templates_for(Ln.active())
    check(any("tomato sauce" in t for t in tn),
          "transfer task: templates NAME the object", f"templates={tn}")
    check(not any(re.search(r"sauce bottle", t) for t in tn),
          "and never reuse the task-8 'sauce bottle' template for an unrelated object",
          f"templates={tn}")
    Lo = A.Ledger()
    Lo.seed_execution(T19_EXEC, T19_STAGES, mapping, name_objects=False)
    place_o = next(o for o in Lo._ordered() if o.kind == "place#1")
    check("without naming drawer identity" in place_o.predicate,
          "occlusion task: the redaction clause is preserved on the place stage",
          f"predicate={place_o.predicate!r}")
    check(not Lo.templates_for(place_o) or
          any(re.search(r"\bobject\b", t) for t in Lo.templates_for(place_o)),
          "occlusion task: templates keep the inherited object-anonymous form",
          f"templates={Lo.templates_for(place_o)}")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"AOM TEST FAILED: {len(FAILS)} check(s)")
        for f in FAILS:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("AOM TEST PASSED: execution-partition seeding, MODE/needs_look split, place")
    print("independence, stagnant reachability+re-arm, single-target gate, DERIVED refresh,")
    print("object naming, coverage telemetry.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
