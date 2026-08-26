from __future__ import annotations

from harness.external_memory import ExternalMemory, EvidenceItem


def memory_search(memory: ExternalMemory, *, query: str, k: int = 6) -> list[EvidenceItem]:
    q = query.lower().strip()
    if not q:
        return list(memory.episode_evidence[-k:])
    scored: list[tuple[int, EvidenceItem]] = []
    for item in memory.episode_evidence:
        hay = " ".join(
            [
                item.action,
                item.instruction,
                item.outcome,
                item.notes,
            ]
        ).lower()
        score = sum(1 for token in q.split() if token in hay)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], -x[1].step))
    if scored:
        return [item for _, item in scored[:k]]
    return list(memory.episode_evidence[-k:])


def memory_reason_for_planner(
    memory: ExternalMemory,
    *,
    stage_name: str | None,
    current_subtask: str,
    stall_steps: int,
    attempt_idx: int,
    candidate_subtask: str | None = None,
) -> str:
    """HarnessVLA-style read-time memory reasoning for the VLM planner (not VLA)."""
    lines: list[str] = []

    if memory.global_rules:
        lines.append("Global memory:")
        for rule in memory.global_rules[:3]:
            lines.append(f"- {rule}")

    if memory.best_partial and memory.best_partial.completed_stages:
        stages = ", ".join(memory.best_partial.completed_stages[-4:])
        lines.append(f"Best prior attempt completed stages: {stages}.")
    if memory.best_partial and memory.best_partial.stage_score_pct > 0:
        lines.append(
            f"Best attempt stage progress: {memory.best_partial.stage_score_pct:.0f}%. "
            "Use keyframes to recall object states from earlier in the episode."
        )

    hits = memory_search(
        memory,
        query=f"stall {stage_name or ''} {current_subtask}",
        k=4,
    )
    if hits:
        lines.append("Recent episode evidence:")
        for item in hits:
            lines.append(
                f"- step={item.step} action={item.action} subtask={item.instruction} "
                f"outcome={item.outcome or 'n/a'}"
            )

    if stage_name:
        lines.append(f"Current incomplete stage: {stage_name}.")
    if stall_steps > 0:
        lines.append(
            f"Progress stalled for {stall_steps} steps on subtask '{current_subtask}'. "
            "Use historical keyframes to recall earlier object states before choosing the next primitive."
        )
    if candidate_subtask and candidate_subtask != current_subtask:
        lines.append(
            f"Suggested primitive for recovery: '{candidate_subtask}'. "
            "Prefer this if the current primitive is not making stage progress."
        )
    if attempt_idx > 0:
        lines.append(
            "This is a resumed attempt after partial progress. Do not repeat already completed stages."
        )
    return "\n".join(lines).strip()
