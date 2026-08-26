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
    """Map the active stage to the most likely primitive label."""
    if not primitive_labels:
        return None

    if stage_idx < len(stage_specs):
        stage_name = str(getattr(stage_specs[stage_idx], "name", "") or "").lower()
        for label in primitive_labels:
            norm = label.lower().replace("_", " ")
            tokens = [tok for tok in norm.split() if len(tok) > 2]
            if tokens and all(tok in stage_name for tok in tokens[:2]):
                return label
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
