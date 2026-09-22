"""Pytest plugin: re-install the ORIGINAL (pre-fix) fuzzy matcher, in memory only.

Loaded with `-p buggy_tie_plugin` so it patches `harness.stage_mapper` BEFORE the test
modules are collected; each test module's `from harness.stage_mapper import ...` therefore
binds the buggy function. Nothing on disk is touched -- this exists to prove the new tests
actually DISCRIMINATE between the fixed and the defective matcher.

Original logic (as it was on disk when the 15-episode pilot ran, 2026-09-20/21), reproduced
from a direct read of the file:
  * index alignment only when |labels| == |specs|  (never true for the 26 memory tasks)
  * fuzzy: hit-count with a STRICT `>` comparison, so ties go to the earliest label
"""
from __future__ import annotations

from typing import Any


def _buggy_expected_primitive_for_stage(
    *,
    primitive_labels: list[str],
    stage_idx: int,
    stage_specs: list[Any],
    stage_done: dict[str, bool],
    current_subtask: str,
) -> str | None:
    if not primitive_labels:
        return None

    if stage_specs and len(primitive_labels) == len(stage_specs) and 0 <= stage_idx < len(primitive_labels):
        return primitive_labels[stage_idx]

    if stage_idx < len(stage_specs):
        stage_name = str(getattr(stage_specs[stage_idx], "name", "") or "").lower()
        best: str | None = None
        best_score = 0
        for label in primitive_labels:
            norm = label.lower().replace("_", " ")
            tokens = [tok for tok in norm.split() if len(tok) > 2]
            if not tokens:
                continue
            hit = sum(1 for tok in tokens if tok in stage_name)
            if hit >= min(2, len(tokens)) and hit > best_score:
                best = label
                best_score = hit
        if best is not None:
            return best
        if "place" in stage_name:
            for label in primitive_labels:
                if label.lower().startswith("place"):
                    return label
        if "pick" in stage_name or "pour" in stage_name:
            for label in primitive_labels:
                low = label.lower()
                if low.startswith("pick") or "pour" in low:
                    return low

    completed = sum(1 for done in stage_done.values() if done)
    idx = min(max(stage_idx, completed), len(primitive_labels) - 1)
    candidate = primitive_labels[idx]

    if current_subtask and current_subtask in primitive_labels:
        cur_i = primitive_labels.index(current_subtask)
        if cur_i + 1 < len(primitive_labels):
            nxt = primitive_labels[cur_i + 1]
            if current_subtask.lower().startswith("pick") and nxt.lower().startswith("place"):
                return nxt
    return candidate


def pytest_configure(config):  # noqa: ARG001
    import harness.stage_mapper as sm

    sm.expected_primitive_for_stage = _buggy_expected_primitive_for_stage
    # `controller` imported the name at module scope; rebind it too so the end-to-end
    # tests exercise the buggy matcher as well.
    try:
        import harness.controller as ctrl
        ctrl.expected_primitive_for_stage = _buggy_expected_primitive_for_stage
    except Exception:
        pass
    print("\n[buggy_tie_plugin] ORIGINAL pre-fix fuzzy matcher installed (in memory)")
