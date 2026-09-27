#!/usr/bin/env python
"""EvMem-GPM gate: predicates, C-controller, gating, no notes."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MEMEXP_EVMEM", "1")

from memexp_evmem import (  # noqa: E402
    Ledger, known_tools, dispatch, parse_tool_call, predicate_scaffold,
    control_reject_reason, apply_control, looks_like_label_primitive,
    attractor_cluster, public_ordinal,
)

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILS.append(label)
        if detail:
            print(f"         {detail}")


def main() -> int:
    print("=" * 72)
    print("EVMEM-GPM GATE")
    print("=" * 72)

    print("\n[T1] ordinal + predicate (no answer-key drawer)")
    led = Ledger()
    led.seed_conditions(["01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"])
    check([led.claims[c].label for c in led._order] ==
          ["step 1 of 3", "step 2 of 3", "step 3 of 3"], "ordinals")
    preds = [led.claims[c].predicate for c in led._order]
    check("grasp/lift" in preds[0].lower(), "lift predicate", preds[0])
    check("count 1" in preds[1], "pour1 predicate", preds[1])
    check("count 2" in preds[2] or "second" in preds[2].lower(), "pour2 predicate", preds[2])
    d = led.render_digest(mode="fluent")
    check("NEED:" in d and "top drawer" not in d.lower(), "NEED in digest, no top drawer")
    check("microwave" in d.lower() and "Forbidden" in d, "microwave forbidden listed")

    print("\n[T2] C-gate")
    led.set_active_stage("01_Lift_Tomato_Sauce", 0)
    check(control_reject_reason("Open the microwave door.", led), "reject microwave")
    check(control_reject_reason("Tilt to pour sauce.", led), "reject pour before lift")
    check(control_reject_reason("Pour One", led), "reject label")
    check(control_reject_reason("Reach for and grasp the sauce bottle.", led) is None, "allow grasp")
    led.note_verified("01_Lift_Tomato_Sauce", 1)
    led.set_active_stage("02_Pour_One", 2)
    check(control_reject_reason("Pour sauce a second time.", led), "reject second before pour1")
    check(control_reject_reason("Pour the sauce onto the target once.", led) is None, "allow pour1")
    out, rw = apply_control(
        '{"current_primitive":"Open the microwave","keyframe_positions":[]}', led)
    check(rw and "microwave" not in out.lower(), "apply_control rewrites", out)

    print("\n[T3] gating + attractor")
    led2 = Ledger()
    led2.seed_conditions(["01_Lift_Tomato_Sauce", "02_Pour_One"])
    led2.set_active_stage("01_Lift_Tomato_Sauce", 0)
    check(led2.decide_mode(0) == "fluent", "start fluent")
    led2.note_action("Open the microwave door.", 1, 1, 5)
    led2.note_action("Open the microwave door.", 2, 2, 6)
    check(led2.force_evidence or led2.decide_mode(0) == "evidence", "microwave attractor -> evidence")
    check(attractor_cluster("Open microwave") == "microwave", "cluster")

    print("\n[T4] tools / no notes")
    check(known_tools() == {"look", "ls", "cat", "grep"}, "tools")
    t, _ = dispatch(led, "note", {}, on_context=set(), frame_store={})
    check("unknown" in t, "no note")
    check(parse_tool_call('{"tool":"look","id":"c1"}', known_tools())["tool"] == "look", "parse")

    print("\n[T5] drawer predicate scrub")
    p = predicate_scaffold("01_Open_Top_Drawer")
    check("top" not in p.lower() and "open" in p.lower(), "open without top", p)
    check(public_ordinal(0, 9) == "step 1 of 9", "ordinal helper")

    print("\n" + "=" * 72)
    if FAILS:
        print(f"FAILED {len(FAILS)}")
        for x in FAILS:
            print(" -", x)
        return 1
    print("EVMEM-GPM GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
