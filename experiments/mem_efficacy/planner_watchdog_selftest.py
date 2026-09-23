#!/usr/bin/env python3
"""Self-test for the dead-planner watchdog in `ApiMemoryPlanner.infer_sync`.

WHY THIS EXISTS
  Jobs 598082/598083/598084 (2026-09-20) ran 50 episodes per arm against a planner that stopped
  answering at 12:02:13. The relay returned `HTTP 403 Forbidden`; `chat_completions` turns that into
  `None`; `infer_sync` fell through to `self._current_subtask` and the run kept going for another
  ~100 minutes. 46 of 50 episodes per arm were scored from a planner that was silent. The numbers
  were not obviously wrong -- they were HIGHER than the healthy baseline, because re-asserting the
  current subtask suits the VLA better than the planner's meandering.

  The only visible tell was that `api_vlm_trace.jsonl` came out EMPTY, which no reader of a score
  table would ever check. So the guard has to be tested, not assumed: this script proves it fires,
  proves it fires at the configured count and not before, and proves it can be disabled.

WHAT IT CHECKS
  T1  a dead endpoint raises after exactly `PLANNER_MAX_CONSECUTIVE_FAILURES` empty responses
  T2  it does NOT raise before that count
  T3  a single success in between resets the count (so a flaky-but-alive endpoint survives)
  T4  `PLANNER_MAX_CONSECUTIVE_FAILURES=0` disables the guard entirely
  T5  the healthy path is untouched: with a good response the counter stays at 0

USAGE
  .venv/bin/python experiments/mem_efficacy/planner_watchdog_selftest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation_benchmark"))
# `api_vlm_planner` imports `keyframe_selection`, which lives here. The runner puts this on the
# path too; mirroring it is what makes this script test the same module the run actually uses.
sys.path.insert(0, str(ROOT / "evaluation_benchmark" / "openpi_minimal_runtime"))

import numpy as np  # noqa: E402

import harness.api_vlm_planner as mod  # noqa: E402

GOOD = '```json\n{"current_primitive": "open top drawer", "keyframe_position": []}\n```'
FRAMES = [(np.zeros((8, 8, 3), dtype=np.uint8), None)]


class _TaskInfo:
    """Permissive stand-in for the runner's task_info.

    `ApiMemoryPlanner` reads a growing set of fields off `task_info` (`brief_description`,
    `scene_description`, `task_block`, `primitive_labels`, ...), and enumerating them here would
    make this test break every time the planner learns to read one more. Unknown attribute reads
    return "" instead, so the test exercises the WATCHDOG rather than the task schema.
    """

    task_id = 12
    task_block = "open the middle drawer and put the cookies in it"
    brief_description = "open the middle drawer and put the cookies in it"
    scene_description = "a kitchen with a drawer and some cookies"
    primitive_labels = ["open top drawer", "open middle drawer", "place cookies"]

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return ""


def fresh(limit: str):
    """A planner with the API stubbed out, plus a handle to flip the stub's answer."""
    os.environ["PLANNER_MAX_CONSECUTIVE_FAILURES"] = limit
    os.environ.setdefault("PLANNER_API_KEY", "stub-not-used")
    state = {"reply": None}
    mod.infer_primitive_via_api = lambda **kw: state["reply"]  # type: ignore[assignment]
    pl = mod.ApiMemoryPlanner(task_info=_TaskInfo(), system_prompt="stub system prompt")
    pl.api_key = "stub-not-used"
    return pl, state


def drive(pl, n: int):
    """Call infer_sync n times, returning how many calls completed before a raise."""
    done = 0
    for i in range(n):
        try:
            pl.infer_sync(i, FRAMES)
        except RuntimeError as exc:
            return done, exc
        done += 1
    return done, None


def main() -> int:
    failures: list[str] = []

    # T1 + T2: raises exactly at the limit, not before.
    LIMIT = 5
    pl, state = fresh(str(LIMIT))
    done, exc = drive(pl, 40)
    if exc is None:
        failures.append(f"T1: no RuntimeError after 40 empty responses (limit={LIMIT})")
    elif done + 1 != LIMIT:
        failures.append(f"T1: raised after {done + 1} empty responses, expected exactly {LIMIT}")
    else:
        print(f"  T1 PASS  raised after exactly {LIMIT} empty responses")
        print(f"           msg: {str(exc)[:110]}...")

    # T3: one success in the middle resets the run of failures.
    pl, state = fresh(str(LIMIT))
    seq = [None] * (LIMIT - 1) + [GOOD] + [None] * (LIMIT - 1)
    raised = None
    for i, r in enumerate(seq):
        state["reply"] = r
        try:
            pl.infer_sync(i, FRAMES)
        except RuntimeError as e:
            raised = (i, e)
            break
    if raised is not None:
        failures.append(
            f"T3: raised at index {raised[0]} although a success separated the two runs of "
            f"{LIMIT - 1} failures; the counter is not 'consecutive'"
        )
    else:
        print(f"  T3 PASS  a success between {LIMIT - 1} failures resets the count "
              f"(counter={pl._consecutive_api_failures})")

    # T4: 0 disables the guard.
    pl, state = fresh("0")
    done, exc = drive(pl, 40)
    if exc is not None:
        failures.append(f"T4: guard fired with PLANNER_MAX_CONSECUTIVE_FAILURES=0 (at {done + 1})")
    else:
        print(f"  T4 PASS  40 empty responses survive when the guard is disabled")

    # T5: healthy responses never trip it.
    pl, state = fresh(str(LIMIT))
    state["reply"] = GOOD
    done, exc = drive(pl, 40)
    if exc is not None:
        failures.append(f"T5: guard fired on healthy responses at call {done + 1}")
    elif pl._consecutive_api_failures != 0:
        failures.append(f"T5: counter={pl._consecutive_api_failures} after 40 good responses")
    else:
        print(f"  T5 PASS  40 good responses, counter stayed 0")

    print()
    if failures:
        print("  FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  PASSED: the watchdog fires at the limit, resets on success, and can be disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
