from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.api_planner import load_api_key, suggest_subtask_via_api
from harness.config import HarnessConfig, load_global_rules
from harness.external_memory import ExternalMemory
from harness.memory_reason import memory_reason_for_planner
from harness.stage_mapper import expected_primitive_for_stage


@dataclass
class HarnessController:
    config: HarnessConfig
    task_id: int
    task_info: Any
    memory: ExternalMemory
    attempt_idx: int = 0
    last_stage_idx: int = 0
    stall_since_step: int = 0
    stall_triggered: bool = False
    stalled_stage_name: str | None = None
    force_replan: bool = False
    subtask_override: str | None = None
    vlm_context: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _memory_root: Path | None = field(default=None, repr=False)
    _last_subtask: str = ""

    @classmethod
    def create(
        cls,
        task_id: int,
        task_info: Any,
        config: HarnessConfig,
        memory_root: Path | None = None,
    ) -> HarnessController:
        rules = load_global_rules(config.global_rules_path)
        memory = ExternalMemory(task_id=task_id, global_rules=rules)
        ctrl = cls(config=config, task_id=task_id, task_info=task_info, memory=memory)
        ctrl._memory_root = memory_root if config.persist_memory else None
        return ctrl

    def begin_attempt(self, attempt_idx: int) -> None:
        self.attempt_idx = attempt_idx
        self.last_stage_idx = 0
        self.stall_since_step = 0
        self.stall_triggered = False
        self.stalled_stage_name = None
        self.force_replan = False
        self.subtask_override = None
        self.vlm_context = ""
        self.memory.reset_attempt()
        if attempt_idx > 0:
            self.memory.on_retry(attempt_idx, reason="stage_checkpoint_retry")
            if self.config.stage_checkpoint_retry and self.memory.best_partial:
                completed = ", ".join(self.memory.best_partial.completed_stages[-6:])
                self.vlm_context = (
                    "Resume attempt after partial progress.\n"
                    f"Already completed stages: {completed or 'none'}.\n"
                    "Continue from the earliest incomplete stage; do not repeat finished stages."
                )

    def on_subtask_update(self, step: int, subtask: str) -> None:
        self._last_subtask = subtask
        self.memory.on_subtask(step, subtask)

    def on_stage_progress(
        self,
        *,
        step: int,
        stage_idx: int,
        stage_specs: list[Any],
        subtask: str,
    ) -> None:
        if stage_idx > self.last_stage_idx:
            prev_name = stage_specs[self.last_stage_idx].name if self.last_stage_idx < len(stage_specs) else None
            if prev_name is not None:
                self.memory.on_stage_complete(step, prev_name, subtask)
            self.last_stage_idx = stage_idx
            self.stall_since_step = step
            self.stall_triggered = False
            self.stalled_stage_name = None
            self._refresh_vlm_context(stage_idx, stage_specs, subtask, step)
            return

        if stage_idx >= len(stage_specs):
            return

        current_name = stage_specs[stage_idx].name
        stall_steps = max(0, step - self.stall_since_step)
        if stall_steps >= self.config.stall_step_threshold and not self.stall_triggered:
            self.stall_triggered = True
            self.stalled_stage_name = current_name
            self.memory.on_stall(step, current_name, subtask, stall_steps)
            if self.config.force_vlm_replan_on_stall:
                self.force_replan = True
            self._handle_stall(
                step=step,
                stage_idx=stage_idx,
                stage_specs=stage_specs,
                subtask=subtask,
                stage_name=current_name,
                stall_steps=stall_steps,
            )

    def _handle_stall(
        self,
        *,
        step: int,
        stage_idx: int,
        stage_specs: list[Any],
        subtask: str,
        stage_name: str,
        stall_steps: int,
    ) -> None:
        candidate = expected_primitive_for_stage(
            primitive_labels=list(getattr(self.task_info, "primitive_labels", []) or []),
            stage_idx=stage_idx,
            stage_specs=stage_specs,
            stage_done={spec.name: False for spec in stage_specs[:stage_idx]},
            current_subtask=subtask,
        )
        if self.config.api_planner_enable:
            api_key = load_api_key(self.config.api_key_file)
            mem_ctx = memory_reason_for_planner(
                self.memory,
                stage_name=stage_name,
                current_subtask=subtask,
                stall_steps=stall_steps,
                attempt_idx=self.attempt_idx,
                candidate_subtask=candidate,
            )
            api_choice = suggest_subtask_via_api(
                task_block=str(getattr(self.task_info, "task_block", "") or ""),
                stage_name=stage_name,
                current_subtask=subtask,
                primitive_labels=list(getattr(self.task_info, "primitive_labels", []) or []),
                memory_context=mem_ctx,
                api_key=api_key,
                base_url=self.config.api_base_url,
                model=self.config.api_model,
            )
            if api_choice:
                candidate = api_choice

        self._refresh_vlm_context(stage_idx, stage_specs, subtask, step, candidate_subtask=candidate)
        if self.config.subtask_override_on_stall and candidate and candidate != subtask:
            self.subtask_override = candidate
            self.memory.append_evidence(
                step=step,
                action="subtask_override",
                instruction=candidate,
                outcome="planned",
                notes=f"stall_recovery:{stage_name}",
            )

    def _refresh_vlm_context(
        self,
        stage_idx: int,
        stage_specs: list[Any],
        subtask: str,
        step: int,
        candidate_subtask: str | None = None,
    ) -> None:
        if not self.config.inject_vlm_context:
            self.vlm_context = ""
            return
        stage_name = stage_specs[stage_idx].name if stage_idx < len(stage_specs) else None
        stall_steps = max(0, step - self.stall_since_step) if self.stall_triggered else 0
        ctx = memory_reason_for_planner(
            self.memory,
            stage_name=stage_name,
            current_subtask=subtask,
            stall_steps=stall_steps,
            attempt_idx=self.attempt_idx,
            candidate_subtask=candidate_subtask,
        )
        if self.vlm_context:
            ctx = self.vlm_context + "\n\n" + ctx
        self.vlm_context = ctx.strip()

    def get_vlm_context(self) -> str:
        return self.vlm_context.strip()

    def consume_force_replan(self) -> bool:
        if not self.force_replan:
            return False
        self.force_replan = False
        return True

    def consume_subtask_override(self) -> str | None:
        override = self.subtask_override
        self.subtask_override = None
        return override

    def override_vla_prompt(self, base_prompt: str, **_kwargs: Any) -> str:
        # Keep VLA prompt clean by default; pi0.5 is not trained on harness prose.
        if self.config.inject_vla_hints:
            return base_prompt
        return base_prompt

    def on_episode_end(
        self,
        *,
        stage_score_pct: float,
        stage_success: bool,
        failure_reason: str | None,
        stage_done: dict[str, bool],
        run_dir: Path | None = None,
    ) -> None:
        self.memory.record_attempt(
            attempt_idx=self.attempt_idx,
            stage_score_pct=stage_score_pct,
            stage_success=stage_success,
            failure_reason=failure_reason,
            stage_done=stage_done,
            stalled_stage=self.stalled_stage_name,
        )
        self.diagnostics = {
            "harness_attempts": self.attempt_idx + 1,
            "harness_stall_triggered": self.stall_triggered,
            "harness_stalled_stage": self.stalled_stage_name,
            "harness_best_stage_score_pct": (
                self.memory.best_partial.stage_score_pct if self.memory.best_partial else stage_score_pct
            ),
            "harness_api_planner": self.config.api_planner_enable,
        }
        if self._memory_root is not None:
            self.memory.save(self._memory_root / f"task{self.task_id}" / "harness_memory.json")
        if run_dir is not None:
            self.memory.save(run_dir / "harness_memory.json")

    def effective_max_retries(self) -> int:
        if self.task_id in self.config.skip_retry_task_ids:
            return 0
        return self.config.max_episode_retries

    def total_attempts_for_task(self) -> int:
        return 1 + max(0, self.effective_max_retries())

    def should_retry(self, stage_success: bool, stage_score_pct: float = 0.0) -> bool:
        if stage_success:
            return False
        if self.attempt_idx >= self.effective_max_retries():
            return False
        best_pct = (
            self.memory.best_partial.stage_score_pct
            if self.memory.best_partial is not None
            else stage_score_pct
        )
        if self.config.smart_retry and best_pct >= self.config.retry_skip_score_pct:
            return False
        if self.config.retry_require_progress and self.attempt_idx > 0:
            if self.memory.best_partial is not None and stage_score_pct <= self.memory.best_partial.stage_score_pct:
                return False
        if not self.config.stage_checkpoint_retry:
            return True
        return True
