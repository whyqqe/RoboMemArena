from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodicEvent:
    step: int
    event_type: str
    subtask: str = ""
    stage_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    step: int
    action: str
    instruction: str
    outcome: str = ""
    notes: str = ""


@dataclass
class AttemptRecord:
    attempt_idx: int
    stage_score_pct: float
    stage_success: bool
    failure_reason: str | None
    completed_stages: list[str]
    stalled_stage: str | None = None


class ExternalMemory:
    """Out-of-model long-horizon memory (HarnessVLA episode + task memory hybrid)."""

    def __init__(
        self,
        *,
        task_id: int,
        global_rules: list[str] | None = None,
        working_window: int = 12,
        episodic_limit: int = 64,
    ) -> None:
        self.task_id = task_id
        self.global_rules = list(global_rules or [])
        self.working_window = working_window
        self.episodic_limit = episodic_limit
        self.working: deque[tuple[int, str]] = deque(maxlen=working_window)
        self.episodic: list[EpisodicEvent] = []
        self.episode_evidence: list[EvidenceItem] = []
        self.task_trace: list[dict[str, Any]] = []
        self.attempts: list[AttemptRecord] = []
        self.best_partial: AttemptRecord | None = None

    def reset_attempt(self) -> None:
        self.working.clear()

    def on_subtask(self, step: int, subtask: str) -> None:
        subtask = subtask.strip()
        if not subtask:
            return
        if not self.working or self.working[-1][1] != subtask:
            self.working.append((step, subtask))
            self._append_episode(EpisodicEvent(step=step, event_type="subtask", subtask=subtask))
            self.task_trace.append({"step": step, "action": "subtask", "prompt": subtask})
            self.append_evidence(
                step=step,
                action="subtask_update",
                instruction=subtask,
                outcome="active",
            )

    def on_stage_complete(self, step: int, stage_name: str, subtask: str) -> None:
        self._append_episode(
            EpisodicEvent(
                step=step,
                event_type="stage_complete",
                subtask=subtask,
                stage_name=stage_name,
            )
        )
        self.task_trace.append(
            {"step": step, "action": "stage_complete", "stage": stage_name, "prompt": subtask}
        )
        self.append_evidence(
            step=step,
            action="stage_complete",
            instruction=subtask,
            outcome="success",
            notes=stage_name,
        )

    def on_stall(self, step: int, stage_name: str, subtask: str, stall_steps: int) -> None:
        self._append_episode(
            EpisodicEvent(
                step=step,
                event_type="stall",
                subtask=subtask,
                stage_name=stage_name,
                metadata={"stall_steps": stall_steps},
            )
        )
        self.append_evidence(
            step=step,
            action="stall",
            instruction=subtask,
            outcome="stalled",
            notes=f"{stage_name}:{stall_steps}",
        )

    def on_retry(self, attempt_idx: int, reason: str) -> None:
        self._append_episode(
            EpisodicEvent(
                step=-1,
                event_type="retry",
                metadata={"attempt_idx": attempt_idx, "reason": reason},
            )
        )
        self.append_evidence(
            step=-1,
            action="retry",
            instruction="",
            outcome="resume",
            notes=reason,
        )

    def append_evidence(
        self,
        *,
        step: int,
        action: str,
        instruction: str,
        outcome: str = "",
        notes: str = "",
    ) -> None:
        self.episode_evidence.append(
            EvidenceItem(
                step=step,
                action=action,
                instruction=instruction,
                outcome=outcome,
                notes=notes,
            )
        )
        if len(self.episode_evidence) > 128:
            self.episode_evidence = self.episode_evidence[-128:]

    def record_attempt(
        self,
        *,
        attempt_idx: int,
        stage_score_pct: float,
        stage_success: bool,
        failure_reason: str | None,
        stage_done: dict[str, bool],
        stalled_stage: str | None,
    ) -> None:
        completed = [name for name, done in stage_done.items() if done]
        record = AttemptRecord(
            attempt_idx=attempt_idx,
            stage_score_pct=stage_score_pct,
            stage_success=stage_success,
            failure_reason=failure_reason,
            completed_stages=completed,
            stalled_stage=stalled_stage,
        )
        self.attempts.append(record)
        if self.best_partial is None or record.stage_score_pct > self.best_partial.stage_score_pct:
            self.best_partial = record

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.task_id,
            "global_rules": self.global_rules,
            "working": list(self.working),
            "episodic": [asdict(e) for e in self.episodic],
            "episode_evidence": [asdict(e) for e in self.episode_evidence],
            "task_trace": self.task_trace,
            "attempts": [asdict(a) for a in self.attempts],
            "best_partial": asdict(self.best_partial) if self.best_partial else None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_episode(self, event: EpisodicEvent) -> None:
        self.episodic.append(event)
        if len(self.episodic) > self.episodic_limit:
            self.episodic = self.episodic[-self.episodic_limit :]
