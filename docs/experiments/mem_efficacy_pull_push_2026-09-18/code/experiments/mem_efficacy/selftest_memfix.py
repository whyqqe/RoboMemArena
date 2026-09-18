"""mem_efficacy / GATE 0c -- prove the memory-content correction works, before a GPU is used.

WHAT THIS GATE IS FOR
---------------------
`validate_arm.py` CHECK 12 proves the correction is INSTALLED. Installed is not the same as
correct: a module can install a renderer that returns an empty string, or fails to patch one of the
two Planner call sites, or exposes the same defects it was written to remove. The distance between
"the flag is on" and "the behaviour changed in the intended direction, and not in some other
direction" is where this project has repeatedly lost runs, so the corrected functions are executed
here against an archived episode and their output is inspected.

It runs entirely on the CPU with no API calls, so it can run before the GPUs are requested.

WHAT IT DOES NOT CLAIM
----------------------
It does NOT claim the correction improves the score. It cannot: that is what the run is for. It
asserts only that the defects are gone and that the corrected path is what the evaluator will call.
If corrected memory still does not help, that is the result.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    root = os.environ.get("ROOT", "")
    for sub in ("evaluation_benchmark", os.path.join("evaluation_benchmark", "openpi_minimal_runtime")):
        p = os.path.join(root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    # Refuse to report success from an environment where the correction is switched off: the
    # checks below would then be testing the ORIGINAL code and passing by reproducing the defects.
    flag = str(os.environ.get("MEMEXP_MEMFIX_ENABLE", "")).strip().lower()
    if flag not in {"1", "true", "yes", "on", "y", "t"}:
        print("[GATE 0c] MEMEXP_MEMFIX_ENABLE is not set: this environment is not a corrected arm, "
              "so there is nothing to test here", file=sys.stderr)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import memexp_memfix as memfix
    except Exception as exc:  # noqa: BLE001
        print(f"[GATE 0c] could not import memexp_memfix: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 3

    res = memfix.verify_fix()
    raw = res.get("refresh") or {}

    print("[GATE 0c] memory-content correction (defects reproduced, then shown fixed)")
    print(f"  fixture            : diagnosis_590799/fixture_task8_attempt0_harness_memory.json")
    print(f"  original render    : {res.get('before_len')} chars")
    print(f"  corrected render   : {res.get('after_len')} chars")
    print(f"  duplicate rows in original render : {res.get('orig_dup_rows')}")
    print(f"  rows dropped by dedupe            : {res.get('corrected_n_dropped')}")
    if raw:
        ol, cl = raw.get("orig_lens") or [], raw.get("corr_lens") or []
        if ol and cl:
            print(f"  context after 12 refresh calls    : original={ol[-1]} chars, "
                  f"corrected {min(cl)}..{max(cl)} chars")

    checks = res.get("checks") or {}
    failed = sorted(k for k, v in checks.items() if not v)
    for k in sorted(checks):
        print(f"  [{'PASS' if checks[k] else 'FAIL'}] {k}")

    if res.get("error"):
        print(f"[GATE 0c] FAILED: {res['error']}", file=sys.stderr)
        return 4
    if failed:
        print(f"[GATE 0c] FAILED: {len(failed)} check(s): {failed}", file=sys.stderr)
        return 5

    print("[GATE 0c] PASSED: the three content defects are reproduced on the original code and "
          "absent from the corrected path, and the corrected path is the one the evaluator calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
