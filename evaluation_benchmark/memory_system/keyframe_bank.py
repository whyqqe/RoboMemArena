from __future__ import annotations

from typing import List

from keyframe_selection import build_visual_memory


def merge_keyframe_bank(
    *,
    j_hist: List[List[int]],
    t: int,
    recent_window: int,
    cluster_distance: int,
    pinned_steps: List[int],
    salient_steps: List[int],
    bank_max: int,
) -> List[int]:
    """
    MemER-style consolidation: cluster nominated indices, keep stage/salient anchors.
    Hyperparameters (d, bank_max) are optimizable without gradient training.
    """
    base = build_visual_memory(j_hist, t=t, N=recent_window, d=cluster_distance)
    candidates = sorted(set(base) | set(pinned_steps) | set(salient_steps))
    if not candidates:
        return []

    # Re-cluster merged candidates so nearby nominations collapse (MemoryVLA-style merge).
    clusters: List[List[int]] = []
    cur = [candidates[0]]
    for idx in candidates[1:]:
        if idx - cur[-1] <= cluster_distance:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)

    selected: List[int] = []
    pinned_set = set(pinned_steps)
    for cluster in clusters:
        if any(c in pinned_set for c in cluster):
            # Prefer earliest pinned step in cluster (stage boundary).
            pinned_in_c = [c for c in cluster if c in pinned_set]
            selected.append(min(pinned_in_c))
        else:
            mid = len(cluster) // 2
            selected.append(cluster[mid])

    cutoff = t - recent_window + 1
    selected = sorted({k for k in selected if k <= cutoff})
    if bank_max > 0 and len(selected) > bank_max:
        pinned_kept = [k for k in selected if k in pinned_set]
        unpinned = [k for k in selected if k not in pinned_set]
        if len(pinned_kept) >= bank_max:
            selected = sorted(pinned_kept)[-bank_max:]
        else:
            room = bank_max - len(pinned_kept)
            selected = sorted(pinned_kept) + unpinned[-room:]
    return selected
