#!/usr/bin/env python
"""mem_efficacy / FUTILITY-gate test: the early-stop DECISION, extracted from the runner itself.

WHY THIS EXISTS. `FUTILITY_STOP_TRIALS` is the only mechanism in this project that kills a running
evaluation on the basis of a SCORE. A false fire destroys a cell's remaining trials, and in the
results it is indistinguishable from a task the arm genuinely fails -- so a bug in the rule looks
exactly like a scientific finding. That is not hypothetical here:

  * job 614798 killed the task-8 treatment at trial 13 under the all-zero rule while one seed was
    scoring 100% on four trials;
  * the FIRST version of the mean rule compared an n-trial running mean against the control's
    FINAL mean. Replayed over the ten archived nomem cells it fires on FOUR of them, and it killed
    live job 614847 (task 19) at n=9 with mean 14.8 while the CONTROL CELL FOR THAT SAME SEED
    finished at 13.32 -- i.e. it stopped the arm for beating the control's own final number.

Both were caught by replaying the rule against archived controls, which is why T7 below is the
most important test in this file. The rules live INSIDE `run_26x1.sbatch`, so this test EXTRACTS
the awk programs from the runner's own source and executes those exact strings: a copy pasted here
would drift the first time the runner is edited, and a rewrite of the runner fails loudly instead.

WHAT IS ASSERTED
  T1  the summary awk recovers trials / max / mean (mean over non-empty rows)
  T2  the all-zero rule fires on all-zero only
  T3  the absolute-floor rule fires at or below the floor, never above
  T4  nothing fires before FUTILITY_TRIALS trials are scored
  T5  floor=0 disables the mean rule and leaves the all-zero rule armed
  T6  fires on the shapes that SHOULD stop, and not on the live near-parity results
  T7  REPLAY: no archived nomem control cell can trip either rule at ANY prefix length

Costs milliseconds and no API quota.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_26x1.sbatch")
RESULTS = os.path.join(HERE, "results")

def _cells(pattern: str) -> list[str]:
    """Absolute glob under RESULTS. Absolute, because a relative glob resolves against the CWD and
    silently returns [] when the test is run from anywhere else -- which would turn T7's most
    important cases into SKIPs that still print PASS."""
    return sorted(glob.glob(os.path.join(RESULTS, pattern)))


# Archived nomem cells, and the calibrated floor each task is run with. Used by T7.
CONTROLS = {
    "5": (_cells("task5_h0_controls/h0/nomem_s*/prompt_trace.tsv"), 5),
    "8": (_cells("hard3_t8_nomem_h0_3x10/h0/nomem_s*/prompt_trace.tsv"), 0),
    "19": (_cells("hard3_t19_nomem_h0_3x10/h0/nomem_s*/prompt_trace.tsv"), 8),
    "22": (_cells("hard3_t22_nomem_h0_3x10/h0/nomem_s*/prompt_trace.tsv"), 25),
}
# The window the runners arm. T7 replays with this and with the strictest window the rule allows.
WINDOW = 6

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail and not ok:
        print(f"         {detail}")
    if not ok:
        FAILS.append(label)
    return ok


def _extract(src: str, start: str, end: str, what: str) -> str:
    i = src.find(start)
    if i < 0:
        raise SystemExit(
            f"FATAL: could not find the {what} anchor {start!r} in run_26x1.sbatch. The runner was "
            f"rewritten; this test must be updated WITH it, not deleted. A futility rule that no "
            f"test can reach is how the last two false stops got through."
        )
    i += len(start)
    j = src.find(end, i)
    if j < 0:
        raise SystemExit(f"FATAL: could not find the end of the {what} ({end!r}).")
    return src[i:j]


def load_programs() -> tuple[str, str]:
    src = open(RUNNER, encoding="utf-8").read()
    summary = _extract(src, "awk -F'\\t' '", "' \"${_ft}\"", "trace-summary awk")
    # The verdict awk's anchor is the PROGRAM OPENER, taken as the ` '` that closes the last `-v`
    # argument. Anchoring on a `-v` line instead put the remaining `-v` arguments inside the
    # extracted program -- valid text for a human, a syntax error for awk. The argument list grew a
    # `-v cz=` / `-v ser=` pair when the consecutive-zero rule was added; anchoring on the program
    # opener keeps this test reaching the rule that was just added, which is the whole point. A
    # rule no test can reach is exactly how the two false stops documented above got through.
    verdict = _extract(src, "-v cz=\"${_futile_consec}\" -v ser=\"${_fseries}\" '", "')", "verdict awk")
    return summary, verdict


def run_summary(program: str, rows: list[float | None]) -> tuple[int, float, float, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        fh.write("task_id\ttrial\tseed\tvlm_ckpt\tvla_prompt_last\tstage_success\tgoal_success\tstage_score_pct\n")
        for i, v in enumerate(rows):
            s = "" if v is None else f"{v}"
            fh.write(f"8\t{i}\t{100 + i}\tx\t\t0\t0\t{s}\n")
        path = fh.name
    try:
        out = subprocess.run(["awk", "-F", "\t", program, path],
                             capture_output=True, text=True, check=True).stdout.split()
    finally:
        os.unlink(path)
    ser = out[3] if len(out) > 3 else ""
    return int(out[0]), float(out[1]), float(out[2]), ser


def run_verdict(program: str, n: float, mx: float, mn: float, fl: float, w: float,
                cz: float = 0, ser: str = "") -> str:
    return subprocess.run(
        ["awk", "-v", f"n={n}", "-v", f"mx={mx}", "-v", f"mn={mn}",
         "-v", f"fl={fl}", "-v", f"w={w}", "-v", f"cz={cz}", "-v", f"ser={ser}", program],
        capture_output=True, text=True, check=True).stdout.strip()


def _read_scores(path: str) -> list[float]:
    out: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 8:
                try:
                    out.append(float(p[7]))
                except ValueError:
                    pass
    return out


def main() -> int:
    print("=" * 78)
    print("FUTILITY GATE: must fire when, and only when, the arm is structurally failing")
    print("=" * 78)
    summary, verdict = load_programs()
    print(f"  summary awk: {summary.strip()[:66]}...")
    print(f"  verdict awk: {verdict.strip().splitlines()[-2].strip()[:66]}")

    print("\n[T1] the summary awk recovers trials / max / mean / series")
    n, mx, mn, ser = run_summary(summary, [0.0, 100.0, 0.0, 50.0])
    check((n, mx, mn) == (4, 100.0, 37.5), "n / max / mean over four trials",
          f"got n={n} max={mx} mean={mn} want 4/100.0/37.5")
    check(ser == "0,100,0,50", "series is the scores in TRIAL ORDER, comma-separated",
          f"got {ser!r} want '0,100,0,50'")
    n, mx, mn, ser = run_summary(summary, [100.0] * 4)
    check((n, mx, mn) == (4, 100.0, 100.0), "all-perfect trace", f"got {n},{mx},{mn}")
    n, mx, mn, ser = run_summary(summary, [])
    check((n, mx, mn) == (0, 0.0, 0.0), "empty trace is n=0, no ZeroDivision", f"got {n},{mx},{mn}")

    print("\n[T2] the all-zero rule fires on all-zero only")
    check(run_verdict(verdict, 4, 0.0, 0.0, 25.0, 4) == "all_zero",
          "four zeros -> all_zero")
    check(run_verdict(verdict, 6, 0.0, 0.0, 0.0, 6) == "all_zero",
          "all-zero rule is armed even when the mean rule is off")
    check(run_verdict(verdict, 6, 100.0, 25.0, 0.0, 6) == "wait",
          "a trace with one perfect trial is NOT all-zero")

    print("\n[T3] the absolute floor fires at or below it, never above")
    check(run_verdict(verdict, 6, 100.0, 24.0, 25.0, 6) == "below_floor",
          "mean 24 <= floor 25 -> fire")
    check(run_verdict(verdict, 6, 100.0, 25.0, 25.0, 6) == "below_floor",
          "mean exactly AT the floor fires (<=, not <)")
    check(run_verdict(verdict, 6, 100.0, 25.1, 25.0, 6) == "wait",
          "mean just above the floor does not fire")

    print("\n[T4] nothing fires before the window is filled")
    for scored in range(0, WINDOW):
        check(run_verdict(verdict, scored, 0.0, 0.0, 25.0, WINDOW) == "wait",
              f"{scored} trial(s) scored, window {WINDOW} -> wait")
    check(run_verdict(verdict, WINDOW, 0.0, 0.0, 25.0, WINDOW) == "all_zero",
          f"{WINDOW} trial(s) scored -> fires")

    print("\n[T5] floor=0 disables the mean rule only")
    check(run_verdict(verdict, 6, 50.0, 0.0, 0.0, 6) == "wait",
          "floor 0 -> mean rule off, a low-scoring arm survives")
    check(run_verdict(verdict, 6, 0.0, 0.0, 0.0, 6) == "all_zero",
          "floor 0 -> all-zero rule still armed")

    print("\n[T6] fires on the shapes that should stop, not on live near-parity results")
    cases = [
        (0.0, 25.0, "below_floor", "all-zero-equivalent low arm -> stop"),
        (12.0, 25.0, "below_floor", "task 22 style: far under its floor -> stop"),
        (23.33, 8.0, "wait", "task 19 actual ER result is ABOVE floor 8 -> must NOT stop"),
        (14.8, 8.0, "wait", "the live t19 stop (mean 14.8, floor would be 8) -> must NOT stop"),
        (33.3, 25.0, "wait", "the live t22 stop (mean 33.3) is above floor 25 -> must NOT stop"),
        (38.47, 0.0, "wait", "task 8 has NO mean rule, so it survives -> must NOT stop"),
    ]
    for mn, fl, want, why in cases:
        v = run_verdict(verdict, 10, 100.0, mn, fl, 6)
        check(v == want, why, f"mean={mn} floor={fl}: got {v!r} want {want!r}")

    print("\n[T7] REPLAY: no archived control cell can trip either rule at ANY prefix")
    print("     (this is the test that would have caught BOTH false stops above)")
    any_cell = False
    for task, (files, floor) in sorted(CONTROLS.items()):
        if not files or not os.path.exists(os.path.join(RESULTS, files[0])):
            print(f"  [SKIP] task {task}: archived control not present")
            continue
        any_cell = True
        fired = []
        for rel in files:
            scores = _read_scores(os.path.join(RESULTS, rel))
            s = 0.0
            for k in range(1, len(scores) + 1):
                s += scores[k - 1]
                v = run_verdict(verdict, k, max(scores[:k]), s / k, floor, WINDOW)
                if v != "wait":
                    fired.append(f"{os.path.basename(os.path.dirname(rel))}@{k}:{v}")
                    break
        check(
            not fired,
            f"task {task}: control never trips the gate (floor {floor})",
            f"FALSE FIRES: {fired}",
        )
    check(any_cell, "at least one archived control was replayed")

    print("\n[T8] the consecutive-zero rule fires on a stalled TAIL, and only on a stalled tail")
    check(run_verdict(verdict, 6, 100.0, 0.0, 0.0, 6, cz=3, ser="100,0,0,0,0,0") == "consec_zero",
          "trailing run of 3 zeros -> consec_zero")
    check(run_verdict(verdict, 6, 100.0, 50.0, 0.0, 6, cz=3, ser="0,0,0,100,100,100") == "wait",
          "three zeros at the START, recovered since -> wait (the tail is what matters)")
    check(run_verdict(verdict, 6, 100.0, 33.3, 0.0, 6, cz=3, ser="0,0,100,0,0,66.7") == "wait",
          "only two trailing zeros -> wait")
    check(run_verdict(verdict, 6, 100.0, 25.0, 0.0, 6, cz=0, ser="0,0,0,0,0,0") == "wait",
          "cz=0 disables the rule even on six zeros")
    check(run_verdict(verdict, 6, 0.0, 0.0, 0.0, 6, cz=3, ser="0,0,0,0,0,0") == "all_zero",
          "precedence: an all-zero arm is reported as all_zero, not consec_zero")
    check(run_verdict(verdict, 6, 100.0, 10.0, 0.0, 6, cz=3, ser="100,0,0,0,0,0") == "consec_zero",
          "fl=0 (task 8 has no mean rule) does not disable the consecutive rule")
    check(run_verdict(verdict, 5, 100.0, 0.0, 0.0, 6, cz=3, ser="0,0,0,0,0") == "wait",
          "window not yet full -> wait even with five trailing zeros")

    print("\n[T8b] SAFETY: the rule cannot trip the task-8 GPM baseline (this is the calibration)")
    print("      cz=3 is the default BECAUSE cz=2 trips the baseline; both halves are asserted,")
    print("      so a future edit that loosens the rule is caught here rather than in a live job.")
    gpm8 = _cells("evmem_gpm_t8_evmem_h0_1x10/h0/*/prompt_trace.tsv")
    if not gpm8 or not os.path.exists(gpm8[0]):
        print("  [SKIP] the task-8 GPM baseline trace is not present in results/")
    else:
        scores = _read_scores(gpm8[0])
        for cz, must_trip in ((3, False), (2, True)):
            fired = None
            for k in range(1, len(scores) + 1):
                ser = ",".join(f"{x:g}" for x in scores[:k])
                if run_verdict(verdict, k, max(scores[:k]), sum(scores[:k]) / k, 0.0, WINDOW,
                               cz=cz, ser=ser) == "consec_zero":
                    fired = k
                    break
            if must_trip:
                check(fired is not None,
                      f"cz=2 DOES trip the task-8 GPM baseline at k={fired} -- which is exactly why"
                      f" the default is 3",
                      f"scores={scores}")
            else:
                check(fired is None,
                      f"cz=3: the task-8 GPM baseline never trips "
                      f"(scores {[round(x, 1) for x in scores]})",
                      f"FALSE FIRE at k={fired}")

    print("\n[T9] REPLAY of the consecutive-zero rule, REPORTED not asserted")
    print("      A stricter rule is allowed to stop archive cells that only LOOKED doomed. What is")
    print("      not allowed is for that set to include the cell a task-8 treatment is compared to,")
    print("      which T8b asserts. Everything else is printed so the cost is visible, not hidden.")
    tripped: list[str] = []
    for pattern in sorted({os.path.dirname(p) for p in _cells("*/*/*/prompt_trace.tsv")}):
        pieces = pattern.split("/")
        cell = f"{pieces[-3]}/{pieces[-2]}" if len(pieces) >= 3 else pattern
        scores = _read_scores(os.path.join(pattern, "prompt_trace.tsv"))
        if len(scores) < WINDOW:
            continue
        for k in range(WINDOW, len(scores) + 1):
            ser = ",".join(f"{x:g}" for x in scores[:k])
            if run_verdict(verdict, k, max(scores[:k]), sum(scores[:k]) / k, 0.0, WINDOW,
                           cz=3, ser=ser) == "consec_zero":
                final = sum(scores) / len(scores)
                tripped.append(f"{cell}@{k} mean@={sum(scores[:k]) / k:.1f} final={final:.1f}")
                break
    print(f"  cz=3, w={WINDOW}: {len(tripped)} archived cell(s) would have stopped early")
    for t in tripped:
        print(f"    - {t}")

    print("\n[T10] THIS RUNNER'S OWN parameters vs the archived SAME-SEED control")
    print("      T7 replays at w=6; this run arms the gate at w=3 with its own floor, and a floor set")
    print("      ABOVE a control's early floor kills the control too. That is not hypothetical: AOM v4")
    print("      ran FLOOR=15 and was stopped at k=9 with mean 14.8 -- while `hard3_t19_nomem_s100`")
    print("      passes through mean 14.8 at k=9 and 13.3 at k=10. The gate fired on the control's own")
    print("      trajectory, so v4's 14.8 is a truncation and not a measurement. This test is what")
    print("      makes that class of mis-set parameter impossible to submit again.")
    watch_u = int(os.environ.get("FUTILITY_WATCH_UNDER_TEST", "3") or 3)
    floor_u = float(os.environ.get("FUTILITY_FLOOR_UNDER_TEST", "8") or 8)
    cz_u = int(os.environ.get("FUTILITY_CONSEC_UNDER_TEST", "3") or 3)
    task_u = str(os.environ.get("FUTILITY_TASK_UNDER_TEST", "19") or "19")
    seed_u = str(os.environ.get("FUTILITY_SEED_UNDER_TEST", "100") or "100")

    ctrl = _cells(f"hard3_t{task_u}_nomem_h0_3x10/h0/nomem_s{seed_u}/prompt_trace.tsv")
    if not ctrl:
        ctrl = _cells(f"*t{task_u}_nomem*/h0/nomem_s{seed_u}*/prompt_trace.tsv")
    if not ctrl:
        check(False, f"archived same-seed control for task {task_u} seed {seed_u} is present",
              "cannot validate the gate against a control that is not in results/")
    else:
        scores = _read_scores(ctrl[0])
        n = len(scores)
        fired = None
        for k in range(1, n + 1):
            if k < watch_u:
                continue
            ser = ",".join(f"{x:g}" for x in scores[:k])
            v = run_verdict(verdict, k, max(scores[:k]), sum(scores[:k]) / k, floor_u, watch_u,
                            cz=cz_u, ser=ser)
            if v != "wait":
                fired = (k, v)
                break
        check(fired is None or fired[0] >= n,
              f"the control is NOT killed before its final trial "
              f"(w={watch_u}, floor={floor_u}, cz={cz_u}, n={n})",
              f"FALSE STOP at k={fired[0]} ({fired[1]}) on the same-seed control; "
              f"scores={[round(x, 1) for x in scores]}")
        # The arming floor is the parameter that broke v4, so assert the relation that matters
        # rather than the number: the floor must sit BELOW the control's minimum running mean at
        # any prefix the gate can see.
        mn_run = min(sum(scores[:k]) / k for k in range(watch_u, n + 1))
        check(floor_u <= 0 or floor_u < mn_run,
              f"floor {floor_u} is below the control's minimum running mean {mn_run:.2f} "
              f"(minimised over the prefixes the gate can see, k>={watch_u})",
              f"floor {floor_u} >= control early floor {mn_run:.2f}: this floor kills the control")

        # What the rule would do to every archived cell of this task: it must stop only cells that
        # are already dead (final mean at or below the control's), not cells that recovered.
        stopped = []
        for p in sorted(set(_cells(f"*t{task_u}_*/h0/*/task19/prompt_trace.tsv")) |
                        set(_cells(f"*t{task_u}_*/h0/*/prompt_trace.tsv"))):
            sc = _read_scores(p)
            if len(sc) < watch_u:
                continue
            mine = os.path.relpath(p, RESULTS).replace("/prompt_trace.tsv", "")
            for k in range(watch_u, len(sc) + 1):
                ser = ",".join(f"{x:g}" for x in sc[:k])
                if run_verdict(verdict, k, max(sc[:k]), sum(sc[:k]) / k, floor_u, watch_u,
                               cz=cz_u, ser=ser) != "wait":
                    stopped.append((mine, k, round(sum(sc[:k]) / k, 1), round(sum(sc) / len(sc), 1),
                                    len(sc)))
                    break
        early = [s for s in stopped if s[1] < s[4]]
        print(f"  archived task-{task_u} cells stopped before their final trial: {len(early)}")
        for mine, k, m_at, m_fin, _n in early:
            print(f"    - {mine} @k={k}  mean@stop={m_at}  final={m_fin}")
        print("  (cells stopping only on the final trial cost nothing and are not listed)")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"FUTILITY GATE TEST FAILED: {len(FAILS)} check(s)")
        for f in FAILS:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("FUTILITY GATE TEST PASSED: fires only after the window fills; the calibrated rules")
    print("(all-zero, below-floor) can never fire on any archived control cell; and the stricter")
    print("consecutive-zero rule cannot fire on the task-8 GPM baseline the treatment is compared to.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
