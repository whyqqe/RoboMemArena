#!/usr/bin/env python3
"""GATE 0 for the cross-task experience memory (RSI treatment): does it FIRE, and is it INERT?

WHAT THIS GATE IS GUARDING AGAINST
  Two defects, and this project has already paid for both.

  1. THE MECHANISM RAN AND DID NOTHING. Job 586700's keyframe channel produced an all-empty
     `J_hist` and the arm still scored, so the "memory" arm was a second baseline wearing a
     treatment's name. For a TREATMENT that is worse than a crash: a crash costs a job, a silent
     no-op costs a conclusion. So this gate does not check that the code is reachable -- it drives
     a REAL `HarnessController` and asserts that a non-empty block comes out, and prints the
     counters that would expose a bank which failed to load.

  2. THE MECHANISM LEAKED INTO THE CONTROL. Every memory delta here is measured against `nomem`,
     so one stray character in the baseline's prompt invalidates the comparison and does it
     invisibly. The check is therefore two-sided: the SAME probe asserts "" for the baseline and
     non-empty for the treatment, at the same inputs.

WHY IT DRIVES A REAL CONTROLLER AND NOT THE FORMATTER
  `render_cross_task_context` is only half the path. The other half is the hook in
  `on_stage_progress` -- and that hook's exact POSITION is the subtle part, because the stage
  handler has early-return branches (stage advance, evidence gate). A formatter that works
  perfectly while the controller never calls it is precisely defect 1. So the probe constructs the
  controller the way the evaluator does, walks stage-by-stage, and reads the context back through
  `compose_planner_context` -- the same function the Planner's prompt is built from.

WHAT IT CANNOT CHECK
  Whether the entry helps. That is a question for a rollout, not a preflight. This gate only
  establishes that the treatment is installed, fires at the intended moment, stays silent before
  it, and that the rendered text does not contain the one string the channel-A regression traces
  to.

USAGE
  python experiments/mem_efficacy/volo_ltm_gate0.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "experiments" / "mem_efficacy" / "arms"

# The exact string the -13.5 pp channel-A result traces to (job 593253): the block named the
# current incomplete stage, and the Planner re-emitted that stage's primitive while its stall
# counter climbed 1 -> 9. The treatment entry is retrieved BY stage but must never disclose it,
# and "incomplete stage" is how the harness phrases that disclosure, so the assertion is on the
# phrase rather than on a stage name -- a stage name is task-specific and a match on it would
# only cover the task it was written for.
FORBIDDEN = ("incomplete stage", "stalled for", "suggested primitive")


def arm_env(arm: str) -> dict[str, str]:
    """Resolve an arm's environment the way the runner does: source it in a clean bash."""
    cmd = f'set -a; ROOT={ROOT} source {ARMS / (arm + ".sh")} >/dev/null 2>&1; env'
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, cwd=str(ROOT))
    env = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env


def probe(arm: str) -> dict:
    """Drive a real HarnessController through an opening stage and read the context back."""
    env = arm_env(arm)

    # Purge the prefixes the arms control from THIS process before applying the arm's values.
    # Without it the three probes share one process and arm N+1 inherits whatever arm N left in
    # `os.environ` -- the exact defect `nomem.sh` purges for, and here it would show up as a
    # FALSE PASS: `nomem` probed last would inherit `HARNESS_CROSS_TASK_LTM=1` from `shamltm`
    # and its non-empty context would be read as a leak.
    for k in [k for k in os.environ
              if k.startswith(("HARNESS_", "MEM_", "MEMEXP_", "PMH_", "VLM_", "N_RECENT",
                               "K_MAX", "D_MERGE"))]:
        del os.environ[k]
    for k, v in env.items():
        os.environ[k] = v

    sys.path.insert(0, str(ROOT / "evaluation_benchmark"))
    # Reload the arm-dependent harness modules so that (a) the bank cache cannot serve one arm the
    # other's entries, and (b) `ltm_stats()` reads the counters of the module instance that
    # actually rendered. Both were wrong in the first version of this gate, and in opposite
    # directions: the cache would have made the sham indistinguishable from the treatment (a false
    # PASS), while the counters -- imported fresh but incremented in the module the already-imported
    # `harness.controller` still held -- reported 0 after a successful injection (a false FAIL).
    # `harness.controller` has to be dropped too, not just the leaf: it binds
    # `render_cross_task_context` at import, so deleting only the leaf leaves the controller
    # calling the old instance.
    for mod in [m for m in list(sys.modules)
                if m.startswith(("harness.cross_task_ltm", "harness.controller",
                                 "harness.belief_contract_memory"))]:
        del sys.modules[mod]

    from types import SimpleNamespace
    from harness.cross_task_ltm import ltm_stats
    from harness.belief_contract_memory import compose_planner_context
    from harness.config import load_harness_config
    from harness.controller import HarnessController

    result: dict = {"arm": arm, "probes": []}
    # Two task_info doubles: one from the family the entry was MINED from, one from a family it
    # must NOT reach. The negative case is the point -- it is what turns "the block appears" into
    # "the block appears WHERE IT IS SUPPOSED TO", and it is the difference between a scoping
    # mechanism and an always-on prompt block. The blocks are the config's own `task_block` text,
    # copied verbatim from
    # `evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json`.
    BLOCKS = {
        "occlusion": ("This is an occlusion task: open the middle drawer, place cookies into the "
                      "middle drawer, then place butter into the middle drawer."),
        "occluded-location": ("This is an occluded-location memory task: open the microwave, place "
                              "cream into the microwave, then place popcorn into the microwave."),
        "counting": ("This is a counting task: first put chocolate into the frypan, then pick up "
                     "the bread."),
    }
    ti = SimpleNamespace(
        task_id=12, task_block=BLOCKS["occlusion"], primitive_labels=["open drawer"],
        task_name="t12", scene_description="sd", brief_description="bd",
    )

    for block, stage_name, steps in (
        ("occlusion", "01_Open_Middle_Drawer", 0),
        ("occlusion", "01_Open_Middle_Drawer", 100),
        ("occlusion", "01_Open_Middle_Drawer", 239),
        ("occlusion", "01_Open_Middle_Drawer", 240),
        ("occlusion", "01_Open_Middle_Drawer", 600),
        ("occlusion", "02_Place_Cookies_Middle_Drawer", 900),
        # A DIFFERENT family, reached only through the tag: this is the cross-task half of the
        # claim, and it is a separate probe rather than the same one relabelled because the task
        # block changes which entries are applicable.
        ("occluded-location", "01_Open_Microwave", 500),
        ("counting", "01_Open_Middle_Drawer", 600),   # NEGATIVE: must NOT receive the entry
        ("counting", "02_Pour_One", 900),
    ):
        try:
            cfg = load_harness_config()
            ti_probe = SimpleNamespace(
                task_id=12, task_block=BLOCKS[block], primitive_labels=["open drawer"],
                task_name="t12", scene_description="sd", brief_description="bd",
            )
            ctrl = HarnessController.create(12, ti_probe, cfg, memory_root=None)
            spec = SimpleNamespace(name=stage_name)
            ctrl.begin_attempt(0)
            # Two calls, and the FIRST is not redundant: `on_stage_progress` advances
            # `last_stage_idx` from -1 to 0 and returns early, so a single call would measure the
            # stage-advance branch instead of the steady-state path the trigger lives on. Between
            # them `stall_since_step` is re-zeroed to model a stage that began at step 0, which is
            # what makes `stall_steps == steps` below.
            ctrl.stall_since_step = 0
            ctrl.on_stage_progress(step=steps, stage_idx=0, stage_specs=[spec],
                                   subtask="open_middle_drawer")
            ctrl.stall_since_step = 0
            ctrl.on_stage_progress(step=steps, stage_idx=0, stage_specs=[spec],
                                   subtask="open_middle_drawer")
            ctx = compose_planner_context(harness=ctrl, bcm=None)
            result["probes"].append({
                "block": block, "stage": stage_name, "steps_on_stage": steps, "len": len(ctx),
                "text": ctx, "err": None,
            })
        except Exception as exc:  # noqa: BLE001 -- a probe failure is a finding, not a crash
            result["probes"].append({
                "block": block, "stage": stage_name, "steps_on_stage": steps, "len": -1,
                "text": "", "err": f"{type(exc).__name__}: {exc}",
            })
    result["stats"] = ltm_stats()
    return result


def main() -> int:
    print("=" * 100)
    print("GATE 0 - cross-task experience memory (RSI treatment)")
    print("=" * 100)
    results = {a: probe(a) for a in ("nomem", "rsimem", "shamltm")}
    fails: list[str] = []

    for arm, r in results.items():
        print(f"\n  --- {arm}")
        for p in r["probes"]:
            tag = f"len={p['len']:>4}" if p["len"] >= 0 else f"ERROR {p['err']}"
            print(f"      {p['block']:18s} {p['stage']:32s} steps={p['steps_on_stage']:>4} {tag}")
        s = r["stats"]
        print(f"      stats: enabled={s['enabled']} loaded={s['counters']['loaded']} "
              f"load_error={s['counters']['load_error']} fired={s['counters']['fired']} "
              f"failed={s['counters']['failed']}")

    # ---- 0. a probe that raised is neither a leak nor a dead treatment -----------------------
    # Ordering matters. The first version of this gate read `len == -1` (its own error sentinel)
    # as "non-zero", and so reported five findings -- LEAK, DEAD TREATMENT, EARLY FIRE, WRONG
    # STAGE, SHAM DEAD -- whose single actual cause was one wrong constructor call. A gate that
    # invents four misleading findings from one real one costs more than the bug it was meant to
    # catch, because the four are what gets debugged first.
    errs = [(a, p) for a, r in results.items() for p in r["probes"] if p["len"] < 0]
    if errs:
        print(f"\n  GATE 0 COULD NOT RUN: {len(errs)} probe(s) raised")
        for a, p in errs:
            print(f"    {a:8s} {p['block']:18s} {p['stage']:32s} steps={p['steps_on_stage']:>4}  {p['err']}")
        print("\n  Nothing below is interpretable: every check reads a length this probe produced.")
        return 2

    # ---- 1. the baseline must be silent at every probe, including past the threshold ---------
    nz = [p for p in results["nomem"]["probes"] if p["len"] != 0]
    if nz:
        fails.append(f"LEAK: nomem produced a non-empty planner context at {len(nz)} probe(s): "
                     f"{[(p['stage'], p['steps_on_stage'], p['len']) for p in nz]}. The control "
                     f"that every memory delta is measured against has moved; no result from this "
                     f"job would be interpretable.")
    else:
        print("\n  [PASS] nomem: planner context empty at every probe")

    # ---- 2. the treatment must fire, and must stay quiet BELOW the threshold -----------------
    r = results["rsimem"]
    by = {(p["block"], p["stage"], p["steps_on_stage"]): p["len"] for p in r["probes"]}
    need = [("occlusion", "01_Open_Middle_Drawer", 240),
            ("occlusion", "01_Open_Middle_Drawer", 600),
            ("occluded-location", "01_Open_Microwave", 500)]
    miss = [k for k in need if by.get(k, 0) <= 0]
    if miss:
        fails.append(f"DEAD TREATMENT: rsimem injected nothing at {miss}. This is the 586700 "
                     f"defect -- an arm configured as a treatment that delivers no treatment. "
                     f"Check load_error in the stats above first: a bank that failed to parse is "
                     f"an ERROR log line and a zero, not a silent skip.")
    else:
        print(f"  [PASS] rsimem fires on the stuck opening stage "
              f"(drawer at 240 -> {by[('occlusion', '01_Open_Middle_Drawer', 240)]}, "
              f"at 600 -> {by[('occlusion', '01_Open_Middle_Drawer', 600)]}, "
              f"microwave at 500 -> {by[('occluded-location', '01_Open_Microwave', 500)]})")

    early = [k for k in (("occlusion", "01_Open_Middle_Drawer", 0),
                         ("occlusion", "01_Open_Middle_Drawer", 100),
                         ("occlusion", "01_Open_Middle_Drawer", 239)) if by.get(k, -1) != 0]
    if early:
        fails.append(f"EARLY FIRE: rsimem injected below the step threshold at {early}. The "
                     f"treatment is supposed to be silent until two attempts have failed; firing "
                     f"sooner makes it a constant block, which is the shape of the channel-A "
                     f"regression (-13.5 pp, job 593253).")
    else:
        print("  [PASS] rsimem is silent below the threshold (0, 100, 239 steps)")

    # Two distinct reasons a probe must stay silent, kept in one check because the required
    # behaviour is identical: a non-opening stage in an applicable family, and the opening stage in
    # a family the entry does not apply to. The second is the cross-task scoping assertion.
    offstage = [k for k in (("occlusion", "02_Place_Cookies_Middle_Drawer", 900),
                            ("counting", "01_Open_Middle_Drawer", 600),
                            ("counting", "02_Pour_One", 900))
                if by.get(k, -1) != 0]
    if offstage:
        fails.append(f"SCOPE VIOLATION: rsimem injected at {offstage}. Either the stage is not a "
                     f"container-opening stage, or the family does not match the entry's declared "
                     f"tags. The block is scoped on both on purpose; an always-on block is the "
                     f"shape of the channel-A regression (-13.5 pp, job 593253).")
    else:
        print("  [PASS] rsimem is scoped: silent on non-opening stages (place/pour) AND on the")
        print("         opening stage of a family its tags exclude (counting)")

    # ---- 3. the rendered text must not contain the channel-A regression's own phrasing --------
    fired = [p for p in r["probes"] if p["len"] > 0]
    bad = [f for f in FORBIDDEN if fired and f in fired[0]["text"].lower()]
    if bad:
        fails.append(f"FORBIDDEN PHRASING: the injected block contains {bad}. That is the "
                     f"disclosure the Planner re-emitted a primitive against. The entry is "
                     f"retrieved BY stage and must not disclose it.")
    else:
        print(f"  [PASS] no stage-disclosure phrasing in the block (checked {list(FORBIDDEN)})")

    # ---- 4. the sham must be a real form control: same trigger, same form, different content ---
    #
    # The assertion is on the FIRE PATTERN, not merely on "it fired somewhere". A sham that fires
    # on different probes is not a control for this treatment: it changes the timing as well as the
    # content, so a delta between them could be either. The first version of the sham bank was
    # scoped to the counting family and was therefore SILENT on every task in this experiment --
    # a second baseline wearing a control's name, which is worse than having no control because the
    # gap looks covered.
    s = results["shamltm"]
    pat = lambda r: {(p["block"], p["stage"], p["steps_on_stage"]) for p in r["probes"] if p["len"] > 0}
    tpat, spat = pat(r), pat(s)
    if not spat:
        fails.append("SHAM DEAD: shamltm injected nothing, so the form control does not exist and "
                     "a treatment effect could not be separated from a form effect.")
    elif tpat != spat:
        fails.append(f"SHAM MISMATCHED: the sham fires on {sorted(spat - tpat)} and misses "
                     f"{sorted(tpat - spat)}. It must fire on exactly the same probes as the "
                     f"treatment, or it controls timing as well as content and a delta between the "
                     f"two arms is uninterpretable.")
    else:
        sfired = [p for p in s["probes"] if p["len"] > 0]
        # Compared on the FULL text, not on a truncation. The two arms share a header by design, so
        # the leading characters are IDENTICAL and a head comparison reports "sham == treatment" for
        # two banks whose bodies differ completely -- which is what the first version of this check
        # did.
        dl = abs(len(sfired[0]["text"]) - len(fired[0]["text"]))
        body_same = (sfired[0]["text"].split("\n", 1)[-1]
                     == fired[0]["text"].split("\n", 1)[-1])
        if sfired[0]["text"] == fired[0]["text"] or body_same:
            fails.append("SHAM == TREATMENT: the two arms inject the same body, so the sham "
                         "controls nothing.")
        else:
            print(f"  [PASS] shamltm fires on EXACTLY the same {len(spat)} probes as the treatment, "
                  f"with different content")
            print(f"         treatment {len(fired[0]['text'])} chars vs sham "
                  f"{len(sfired[0]['text'])} chars (delta {dl})")
            if dl > 40:
                print(f"         NOTE: length delta {dl} chars is larger than expected; the arms "
                      f"are not as closely length-matched as the bank files claim.")

    print()
    print("  " + "-" * 96)
    if fails:
        print(f"  GATE 0 FAILED ({len(fails)} finding(s)):")
        for f in fails:
            print(f"    ! {f.replace(chr(10), ' ')}")
        return 1
    print("  GATE 0 PASSED: the treatment is installed, fires only where intended, is silent where")
    print("  the baseline must be, does not disclose the stage, and has a matched form control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
