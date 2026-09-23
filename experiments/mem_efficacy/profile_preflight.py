#!/usr/bin/env python3
"""Stage-1 profile preflight: prove Hlegacy/H0 overlays land where the plan says they do.

WHY THIS EXISTS
---------------
`profiles/h0.sh` is sourced AFTER an arm. If someone sources it BEFORE the arm, or forgets
it, or an arm re-exports `HARNESS_MAX_RETRIES=2` after the profile, the run silently becomes
Hlegacy while the banner still says H0. That is exactly the class of failure that made the
VoLo t4t5 baseline unpublishable: two writers, one directory, a score that looked fine.

This script does not trust the banner. It sources each (arm, profile) pair in a fresh
subprocess and asserts the resolved env matches the profile contract.

Exit 0 = every declared pair is correct. Exit 1 = at least one pair is wrong; do not submit.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent

H0_MUST = {
    "HARNESS_MAX_RETRIES": "0",
    "HARNESS_SUBTASK_OVERRIDE": "0",
    "HARNESS_STAGE_CHECKPOINT": "0",
    "HARNESS_SMART_RETRY": "0",
    "HARNESS_FORCE_VLM_REPLAN": "0",
    "HARNESS_PERSIST_MEMORY": "0",
    "MEM_STAGE_ANCHOR": "0",
    "HARNESS_CROSS_TASK_LTM": "0",
    "HARNESS_STAGE_LEDGER_GUARD": "0",
    "MEMPROF_NAME": "h0",
}

HLEGACY_MUST = {
    "HARNESS_MAX_RETRIES": "2",
    "HARNESS_SUBTASK_OVERRIDE": "1",
    "HARNESS_STAGE_CHECKPOINT": "1",
    "MEMPROF_NAME": "hlegacy",
}

FORBIDDEN_IN_BOTH = (
    "HARNESS_CROSS_TASK_LTM_BANK",  # must be unset / empty unless an RSI arm is declared
)


def _resolve(arm: str, profile: str) -> dict[str, str]:
    script = f"""
set -euo pipefail
export ROOT="{ROOT}"
source "{EXP}/arms/{arm}.sh"
source "{EXP}/profiles/{profile}.sh"
# Match the PREFIX, then the rest of the key — a trailing `=` on the group would only
# catch bare `HARNESS_=` / `MEM_=` and silently drop every real knob.
env | grep -E '^(HARNESS_|MEM_|MEMEXP_|MEMPROF_|VLM_USE_KEYFRAME_MEMORY|K_MAX|D_MERGE|N_RECENT|PLANNER_|PROACTIVE_MODE)' | sort
"""
    proc = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sourcing {arm}+{profile} failed:\n{proc.stderr[-800:]}")
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k] = v
    return out


def _check(arm: str, profile: str, expect: dict[str, str]) -> list[str]:
    env = _resolve(arm, profile)
    errs: list[str] = []
    for k, want in expect.items():
        got = env.get(k, "<absent>")
        if got != want:
            errs.append(f"{arm}+{profile}: {k}={got!r} want {want!r}")
    # Memory arms under H0 must still keep their write knobs (dense store).
    if profile == "h0" and arm in ("pushmem", "pullmem"):
        if env.get("MEM_KF_STORE_INTERVAL", "0") in ("", "0"):
            errs.append(f"{arm}+h0: MEM_KF_STORE_INTERVAL must stay ON (write program is shared)")
        if env.get("VLM_USE_KEYFRAME_MEMORY") != "1":
            errs.append(f"{arm}+h0: VLM_USE_KEYFRAME_MEMORY must stay 1")
    if profile == "h0" and arm == "nomem":
        if env.get("VLM_USE_KEYFRAME_MEMORY", "0") not in ("0", "<absent>", ""):
            errs.append(f"nomem+h0: VLM_USE_KEYFRAME_MEMORY leaked ON ({env.get('VLM_USE_KEYFRAME_MEMORY')!r})")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["nomem", "pushmem", "pullmem"])
    ap.add_argument("--profiles", nargs="+", default=["hlegacy", "h0"])
    args = ap.parse_args()

    print("=" * 88)
    print("  Stage-1 profile preflight")
    print("=" * 88)
    all_errs: list[str] = []
    for profile in args.profiles:
        expect = H0_MUST if profile == "h0" else HLEGACY_MUST
        for arm in args.arms:
            print(f"  probing {arm} + {profile} ...", flush=True)
            try:
                errs = _check(arm, profile, expect)
            except RuntimeError as exc:
                errs = [str(exc)]
            if errs:
                for e in errs:
                    print(f"    FAIL {e}")
                all_errs.extend(errs)
            else:
                print("    PASS")
    print("-" * 88)
    if all_errs:
        print(f"FAILED: {len(all_errs)} contract violation(s); refusing to submit.")
        return 1
    print("PASSED: every (arm, profile) pair resolves to its contract.")
    return 0


if __name__ == "__main__":
    # Avoid leaking MEMPROF_* from the caller's shell into the subprocesses' inheritance
    # before they source anything — the sourced files must be the sole source of truth.
    for k in list(os.environ):
        if k.startswith("MEMPROF_"):
            del os.environ[k]
    raise SystemExit(main())
