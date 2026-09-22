from __future__ import annotations

from typing import Any


def expected_primitive_for_stage(
    *,
    primitive_labels: list[str],
    stage_idx: int,
    stage_specs: list[Any],
    stage_done: dict[str, bool],
    current_subtask: str,
) -> str | None:
    """Map the active stage to the most likely primitive label.

    Preference order (A1 — granularity alignment):
      1. Index alignment when |labels| == |specs|: BDDL `primitive_order` is designed to be
         1:1 with scored stages. This is the only mapping that correctly handles REPEAT
         stages (07_Open_Top_Drawer_Again → "open top drawer again", not the shorter
         "open top drawer" that a token-prefix matcher would prefer).
      2. Longest fuzzy token match against the stage name (so "open top drawer again"
         beats "open top drawer" when both tokens[:2] match).
      3. Fall back to index-by-completed-count.
    """
    if not primitive_labels:
        return None

    # (1) Index alignment — the measured defect fix. Without this, stage 07 gets
    # "open top drawer" and the VLA has no "again" signal.
    if stage_specs and len(primitive_labels) == len(stage_specs) and 0 <= stage_idx < len(primitive_labels):
        return primitive_labels[stage_idx]

    if stage_idx < len(stage_specs):
        stage_name = str(getattr(stage_specs[stage_idx], "name", "") or "").lower()
        # (2) Longest fuzzy match: score by number of tokens present in the stage name.
        best: str | None = None
        best_score = 0
        for label in primitive_labels:
            norm = label.lower().replace("_", " ")
            tokens = [tok for tok in norm.split() if len(tok) > 2]
            if not tokens:
                continue
            hit = sum(1 for tok in tokens if tok in stage_name)
            # Require at least the first two tokens (legacy behaviour) AND prefer longer.
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
                    return label

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
