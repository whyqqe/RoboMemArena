from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from harness.api_planner import load_api_key, suggest_subtask_via_api
from harness.config import HarnessConfig, load_global_rules
from harness.external_memory import ExternalMemory
from harness.memory_reason import memory_reason_for_planner
from harness.stage_mapper import expected_primitive_for_stage
from harness import redact
from harness.seam_memory import seam_enabled


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int_local(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


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
    # CGMH evidence-gated retry bookkeeping
    evidence_gate_request: bool = False
    evidence_gate_fired: bool = False
    evidence_gate_recovery_at: int = -1
    _deferred_stall: dict[str, Any] | None = field(default=None, repr=False)
    # A5: rolling recovery ladder state (see HarnessConfig.stall_rolling).
    stall_recoveries_here: int = 0
    hard_recovery_request: bool = False
    noop_streak: int = 0
    n_language_noop: int = 0
    n_hard_recovery_request: int = 0
    n_recovery_escalation: int = 0
    n_tv_release: int = 0
    # rho (SEAM): the ladder's input. `seam_store`, when attached, replaces the TEXT-based no-op
    # test in `_after_stall_recovery` with an EVIDENCE-based one.
    #
    # Why this is the highest-value line in the SEAM change: the old test was
    # `subtask_override is None`, i.e. "did the planner say a different string". Measured on
    # 587161, that predicate was true on 176/213 rungs -- and on `03_Pour_Two` it is true
    # BECAUSE THE PRIMITIVE IS CORRECT. The robot must pour again; it must emit the same
    # primitive to do so. So the ladder's trigger was firing hardest exactly where the action was
    # right and the world had not caught up, and it responded by rewriting a string that was
    # already correct. SEAM replaces "did the text change" with "did the EVIDENCE change", which
    # on that stage is the pour count.
    #
    # Held as a store reference rather than a callback on purpose: a callback would have to be
    # installed by whoever builds the planner, and this project has already lost one gate
    # (gate 0b', job 587161) to a proxy that was configured differently from the thing it stood
    # for. A direct reference cannot be mis-wired.
    seam_store: Any = None
    last_rung_kind: str = ""
    _last_rung_sig: tuple[str, int, int] | None = field(default=None, repr=False)
    # rho/lambda (SEAM v2). These are REAL FIELDS, not `diagnostics` keys, and that is a bug fix.
    # The previous arm incremented store-side counters (`n_seam_rung_typed` = 1216) while the
    # controller-side mirror read 0 on every episode, because `diagnostics` is rebuilt on stage
    # advance. A counter that the reporting path cannot see is worse than no counter: the census
    # read 0 and the natural conclusion was "the ladder never fired".
    n_rung_deduped: int = 0
    n_rung_typed: int = 0
    n_gate_fired: int = 0
    n_gate_abstained: int = 0
    n_gate_hard_override: int = 0
    n_gate_cleared_override: int = 0
    n_gate_hint: int = 0
    n_gate_kind_never_confirmed: int = 0
    n_gate_kind_partial_count: int = 0
    n_gate_kind_no_progress: int = 0
    n_gate_kind_unreachable: int = 0
    last_gate_kind: str = ""
    last_gate_blocked_by: str = ""
    _prev_gate_blocked_by: str = field(default="", repr=False)
    _stall_candidate: str = field(default="", repr=False)
    _stall_alternatives: tuple[str, ...] = field(default_factory=tuple, repr=False)

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
        self.evidence_gate_request = False
        self.evidence_gate_fired = False
        self.evidence_gate_recovery_at = -1
        self._deferred_stall = None
        self.stall_recoveries_here = 0
        self.hard_recovery_request = False
        self.noop_streak = 0
        self._prev_gate_blocked_by = ""
        self._last_rung_sig = None
        self.last_gate_kind = ""
        self.last_gate_blocked_by = ""
        self.last_rung_kind = ""
        self._stall_candidate = ""
        self._stall_alternatives = ()
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
            # A5: progress re-arms the whole ladder. Without this the escalation count would
            # carry across stages and a task with many short stages would exhaust its budget
            # on the first one.
            self.stall_recoveries_here = 0
            self.hard_recovery_request = False
            self.noop_streak = 0
            # rho: a new stage is new evidence; keeping the old signature would dedupe the
            # FIRST rung of the next stage against the LAST rung of the previous one and
            # silently deny it a recovery.
            self._last_rung_sig = None
            self.last_rung_kind = ""
            self._prev_gate_blocked_by = ""
            self.last_gate_kind = ""
            self.last_gate_blocked_by = ""
            self.stalled_stage_name = None
            self.evidence_gate_fired = False
            self.evidence_gate_recovery_at = -1
            self._deferred_stall = None
            self._refresh_vlm_context(stage_idx, stage_specs, subtask, step)
            return

        if stage_idx >= len(stage_specs):
            return

        current_name = stage_specs[stage_idx].name
        # SEAM: the stage IN FORCE is not yet an event, but its obligation is live and must be
        # projected. Without this the requirement is only visible once a stage CONFIRMS, i.e.
        # never on the tasks we lose -- `03_Pour_Two` is precisely a stage that never confirms,
        # and its requirement ("two pours") is the one fact the planner needs.
        if self.seam_store is not None:
            try:
                self.seam_store.note_current_stage(current_name)
            except Exception:
                logger.exception("[seam] could not record the live stage")
        stall_steps = max(0, step - self.stall_since_step)

        # Flush deferred recovery after evidence-gate inquire window.
        if (
            self._deferred_stall is not None
            and self.evidence_gate_recovery_at >= 0
            and int(step) >= int(self.evidence_gate_recovery_at)
        ):
            deferred = self._deferred_stall
            self._deferred_stall = None
            self.evidence_gate_recovery_at = -1
            self._handle_stall(**deferred)
            if bool(getattr(self.config, "stall_rolling", False)):
                self._after_stall_recovery()
            self.diagnostics["evidence_gate_flushed"] = int(self.diagnostics.get("evidence_gate_flushed", 0)) + 1

        # A5: rolling ladder (see HarnessConfig.stall_rolling). The historical gate was
        # `not self.stall_triggered`, which latched for the rest of the stage; `_rolling` instead
        # bounds the count and re-arms the clock, so a stage that never progresses gets repeated
        # escalation instead of exactly one offer.
        _rolling = bool(getattr(self.config, "stall_rolling", False))
        _cap = max(1, int(getattr(self.config, "stall_max_recoveries", 3) or 1))
        _armed = (self.stall_recoveries_here < _cap) if _rolling else (not self.stall_triggered)
        if stall_steps >= self.config.stall_step_threshold and _armed:
            self.stall_triggered = True
            if _rolling:
                self.stall_recoveries_here += 1
                # Re-arm the window from NOW, so the next escalation is another full threshold
                # of genuine non-progress away rather than firing on consecutive chunks.
                self.stall_since_step = int(step)
                self.diagnostics["stall_recovery"] = int(
                    self.diagnostics.get("stall_recovery", 0)
                ) + 1
            self.stalled_stage_name = current_name
            self.memory.on_stall(step, current_name, subtask, stall_steps)
            if self.config.force_vlm_replan_on_stall:
                self.force_replan = True

            # Mech-E: defer recovery; optionally arm Inquiry before override.
            if (
                (self.config.evidence_gate or self.config.evidence_gate_blind)
                and not self.evidence_gate_fired
            ):
                self.evidence_gate_fired = True
                self.force_replan = True
                self.evidence_gate_recovery_at = int(step) + int(self.config.evidence_gate_delay_steps)
                self._deferred_stall = {
                    "step": step,
                    "stage_idx": stage_idx,
                    "stage_specs": stage_specs,
                    "subtask": subtask,
                    "stage_name": current_name,
                    "stall_steps": stall_steps,
                }
                if self.config.evidence_gate and not self.config.evidence_gate_blind:
                    self.evidence_gate_request = True
                    outcome = "inquire_before_retry"
                    self.diagnostics["evidence_gate"] = int(self.diagnostics.get("evidence_gate", 0)) + 1
                else:
                    self.evidence_gate_request = False
                    outcome = "blind_delay_before_retry"
                    self.diagnostics["evidence_gate_blind"] = int(
                        self.diagnostics.get("evidence_gate_blind", 0)
                    ) + 1
                self.memory.append_evidence(
                    step=step,
                    action="evidence_gate",
                    instruction=subtask,
                    outcome=outcome,
                    notes=f"delay={self.config.evidence_gate_delay_steps}",
                )
                return

            self._handle_stall(
                step=step,
                stage_idx=stage_idx,
                stage_specs=stage_specs,
                subtask=subtask,
                stage_name=current_name,
                stall_steps=stall_steps,
            )
            if _rolling:
                self._after_stall_recovery()

    def _after_stall_recovery(self) -> None:
        """A5: escalate the ladder, but only once the language rung has really been spent.

        `_handle_stall` sets `subtask_override` ONLY when the stage's expected primitive differs
        from the subtask in force. On t22 (job 582251) the planner had already named the candidate
        ("pick tomato"), so the rung set nothing, yet it consumed the single latch -- the robot then
        had no recovery offer for the remaining ~2400 steps and finished 0/3 stages while HM
        finished 3/3.

        THE OVER-ESCALATION BUG (job 582503). The first version of this method armed the physical
        rung on the FIRST no-override rung, on the reasoning that "the language rung had nothing to
        change". That reasoning is wrong, and the run measured it: `subtask_override is None` does
        NOT mean the language rung did nothing. The same stall branch also raises `force_replan`
        and refreshes the VLM context, so the planner gets a fresh decision from it -- and arming
        the teleport immediately meant the planner never got a chance to answer that replan.
        Measured consequence: all 7 completed tasks spent the FULL 3/3 rewind budget inside stage 0
        (t1: rewinds at t=240/400/560, exactly `after`=160 apart, i.e. saturating on the clock), and
        t1 fell from 100 in every historical local baseline (`local_ctl`, `pmh_md_local`,
        `local_hm`, all 100) to 50. A rewind to the stage start discards whatever partial progress
        the VLA had made, so firing it on a stage that is merely slow is a real cost.

        So the escalation now requires PROOF that language is out of ideas: `noop_before_hard`
        consecutive no-override rungs (default 2), each separated by a full `stall_step_threshold`
        window. Two windows is ~160 steps ~= 16 VLA chunks of genuine non-progress with the planner
        re-deciding in between -- which is the t22 signature (14 identical outputs), and is not the
        signature of a stage that is simply taking a while.
        """
        if self.subtask_override is not None:
            self.n_recovery_escalation += 1
            self.noop_streak = 0
            # A late language win must CANCEL a pending teleport: the rung it was waiting for
            # finally produced something, so the physical rung is no longer the only option left.
            self.hard_recovery_request = False
            self.diagnostics["stall_language_rung"] = int(
                self.diagnostics.get("stall_language_rung", 0)
            ) + 1
            return

        # rho (SEAM): with an evidence store attached, "the language rung was a no-op" is decided
        # by the RESIDUAL, not by the text. See the field comment on `seam_store`.
        if self.seam_store is not None:
            # The gate is fed the SAME candidate `_handle_stall` just computed, so the two cannot
            # disagree about what the stage expects. Feeding it anything else is how a gate ends
            # up gating something other than what actually executes.
            self._after_stall_residual(
                expected_primitive=str(getattr(self, "_stall_candidate", "") or ""),
                alternatives=tuple(getattr(self, "_stall_alternatives", ()) or ()),
            )
            return

        self.n_language_noop += 1
        self.noop_streak = int(getattr(self, "noop_streak", 0)) + 1
        self.diagnostics["stall_language_noop"] = int(
            self.diagnostics.get("stall_language_noop", 0)
        ) + 1
        needed = max(1, int(getattr(self.config, "stall_noop_before_hard", 2) or 1))
        if self.noop_streak < needed:
            self.diagnostics["stall_language_grace"] = int(
                self.diagnostics.get("stall_language_grace", 0)
            ) + 1
            return
        self.hard_recovery_request = True
        self.n_hard_recovery_request += 1

    def hard_recovery_armed(self) -> bool:
        """Sticky: stay armed until a rewind actually lands or the stage advances.

        NOT a one-shot consume: `pact.tick` can legitimately refuse on the same step (its own
        `after` floor, or a clamped gripper). Consuming here would drop the escalation and
        silently reproduce the latch this replaced.
        """
        return bool(self.hard_recovery_request)

    def note_hard_recovery(self, applied: bool) -> None:
        if applied:
            self.hard_recovery_request = False

    def note_tv_release(self) -> None:
        """Count the TV-style forced gripper release issued by the hard rung.

        This used to be a bare local (`recovery_releases`) inside the eval loop, which meant the
        transition-verifier half of A5 was unmeasurable -- the same dead-counter pattern as the
        `n_actuator_emitted_canonical` that had to be deleted for reporting an invariant that was
        never actually tested. Route it through the controller so it rides `recovery_stats()` into
        the durable census.
        """
        self.n_tv_release += 1

    def recovery_stats(self) -> dict[str, Any]:
        out = {
            "n_language_noop": self.n_language_noop,
            "n_language_grace": int(self.diagnostics.get("stall_language_grace", 0)),
            "n_hard_recovery_request": self.n_hard_recovery_request,
            "n_recovery_escalation": self.n_recovery_escalation,
            "n_tv_release": self.n_tv_release,
            "stall_recoveries_here": self.stall_recoveries_here,
            "noop_streak": int(getattr(self, "noop_streak", 0)),
            "hard_recovery_pending": int(bool(self.hard_recovery_request)),
            # rho. `n_rung_deduped` is the reading of the fix: the old ladder counted every
            # text-unchanged firing as an escalation (176/213 on 587161), and a rung whose
            # EVIDENCE is unchanged is now visible as what it always was -- a repeat, not a
            # recovery. `last_rung_kind` names the action the residual selected, so a census can
            # tell "the ladder chose a repeat because the count was short" from "the ladder chose
            # a physical rung", which the previous arm could not distinguish at all.
            # rho/lambda. These read REAL FIELDS now. The previous arm's mirror read the
            # `diagnostics` dict, which is rebuilt on stage advance, so store-side counters showed
            # 1216 typings while every controller-side counter read 0 -- a census that could only
            # conclude "the ladder never fired". `n_gate_hard_override` separately counts the one
            # intervention that actually rewrites the primitive, and
            # `n_gate_cleared_override` counts the SUBTRACTIVE fix: a rung that removed a stale
            # override because repetition was the correct action. A ladder with only additive
            # rungs cannot report that it declined to act.
            "last_rung_kind": str(getattr(self, "last_rung_kind", "")),
            "n_rung_deduped": self.n_rung_deduped,
            "n_rung_typed": self.n_rung_typed,
            "n_gate_fired": self.n_gate_fired,
            "n_gate_abstained": self.n_gate_abstained,
            "n_gate_hard_override": self.n_gate_hard_override,
            "n_gate_cleared_override": self.n_gate_cleared_override,
            "n_gate_hint": self.n_gate_hint,
            "n_gate_kind_never_confirmed": self.n_gate_kind_never_confirmed,
            "n_gate_kind_partial_count": self.n_gate_kind_partial_count,
            "n_gate_kind_no_progress": self.n_gate_kind_no_progress,
            "n_gate_kind_unreachable": self.n_gate_kind_unreachable,
            "last_gate_blocked_by": str(getattr(self, "last_gate_blocked_by", "")),
        }
        return out

    def _after_stall_residual(self, *, expected_primitive: str = "",
                              alternatives: Sequence[str] = ()) -> None:
        """lambda: turn the residual TYPE into a typed harness decision, on EVIDENCE not on text.

        THE PROBLEM THIS REPLACES. The old trigger was `subtask_override is None`, i.e. "did the
        planner say a different string". Measured on 587161 that predicate was true on 176/213
        rungs -- and on an ordinal stage like `03_Pour_Two` it is true BECAUSE THE PRIMITIVE IS
        CORRECT: the robot must pour again, and to do so it must emit the same primitive twice.
        So the ladder fired hardest exactly where the action was right and the world had not
        caught up yet, and responded by rewriting a string that was already correct. A firing
        count dominated by that case is not a measure of recovery.

        THREE BEHAVIOURS, TAKEN FROM HARNESSVLA'S GATE, and each is separately countable:

          never_confirmed  the grader has confirmed 0 outcomes while the trace shows attempts.
                           This is the case where a SOFT hint measured ~10% and a HARD gate
                           measured ~32.2% in HarnessVLA, so it is the ONE case where overriding
                           the primitive is justified. Bounded by `SEAM_GATE_MAX_HARD` so it
                           cannot saturate the way REWIND did (which spent its full budget inside
                           stage 0 on the clock alone; job 582503).
          partial_count    the grader confirmed some but not enough. REPEATING IS CORRECT. The
                           gate says so explicitly AND CLEARS any pending override, because a
                           stale override from an earlier rung is an active instruction to abandon
                           the one action that works.
          no_progress / unreachable  directive only; no override.

        R1/R2/R3: the type decides the action, the clock never does, and a rung that would repeat
        with UNCHANGED evidence is counted as deduped without consuming a recovery. Refusing to
        fire when the obligation is already met is a decision, and it is counted as `abstained`
        rather than being invisible.
        """
        stage = getattr(self, "stalled_stage_name", "") or ""
        try:
            dec = self.seam_store.gate(
                stage, expected_primitive=expected_primitive, alternatives=alternatives
            )
        except Exception:
            logger.exception("[seam] gate failed; falling back to the language accounting")
            self.n_language_noop += 1
            needed = max(1, int(getattr(self.config, "stall_noop_before_hard", 2) or 1))
            if self.noop_streak + 1 >= needed:
                self.hard_recovery_request = True
                self.n_hard_recovery_request += 1
            return

        # The store's own gate already deduped on the evidence rank; this mirrors it controller-side
        # so the two readings can be diffed (a mismatch means one of them is not being called).
        sig = (dec.blocked_by, int(dec.arm_physical))
        if not dec.blocked_by:
            self.n_gate_abstained += 1
            self.last_gate_blocked_by = "abstain"
            return
        if dec.blocked_by == getattr(self, "_prev_gate_blocked_by", ""):
            self.n_rung_deduped += 1
            return
        self._prev_gate_blocked_by = dec.blocked_by
        self.n_rung_typed += 1
        self.last_gate_kind = dec.blocked_by
        self.last_gate_blocked_by = dec.blocked_by
        self.last_rung_kind = dec.blocked_by
        self.n_gate_fired += 1
        for kind in ("never_confirmed", "partial_count", "no_progress", "unreachable"):
            if dec.blocked_by == kind:
                setattr(
                    self, f"n_gate_kind_{kind}", int(getattr(self, f"n_gate_kind_{kind}", 0)) + 1
                )

        if dec.recovery_hint:
            self.n_gate_hint += 1

        if dec.blocked_by == "partial_count":
            # Repetition is the correct action. Clear a pending override rather than setting one.
            if self.subtask_override is not None:
                self.subtask_override = None
                self.n_gate_cleared_override += 1
            self.noop_streak = 0
            self.n_recovery_escalation += 1
            return

        if dec.blocked_by == "never_confirmed":
            hard = _truthy_env("SEAM_GATE_HARD_OVERRIDE", default=True)
            cap = _env_int_local("SEAM_GATE_MAX_HARD", 3)
            if hard and dec.forced_skills and self.n_gate_hard_override < cap:
                cand = str(dec.forced_skills[0]).strip()
                if cand and cand != (self.subtask_override or ""):
                    self.subtask_override = cand
                    self.n_gate_hard_override += 1
                    self.memory.append_evidence(
                        step=-1,
                        action="seam_gate_override",
                        instruction=cand,
                        outcome="blocked_never_confirmed",
                        notes=f"stage={stage}",
                    )
            self.noop_streak = 0
            self.n_recovery_escalation += 1
            return

        if dec.arm_physical:
            self.hard_recovery_request = True
            self.n_hard_recovery_request += 1
            self.n_gate_kind_unreachable += 0
            logger.info("[seam] gate armed the physical rung (blocked_by=%s)", dec.blocked_by)
            return

        # no_progress: a directive with new evidence, no override, no physical rung.
        self.noop_streak = 0
        self.n_recovery_escalation += 1

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
        # Hand the resolved candidate (and the stage's other admissible labels) to rho. The gate
        # must name skills from the SAME vocabulary the stage matcher accepts, otherwise its
        # `forced_skills` are instructions the environment cannot recognise. That desynchronisation
        # is exactly what made the alpha-guard's exit(2) path rewrite a correct primitive.
        self._stall_candidate = str(candidate or "")
        labels = [str(x) for x in (getattr(self.task_info, "primitive_labels", []) or [])]
        self._stall_alternatives = tuple(
            lab for lab in labels if lab and lab != str(candidate or "")
        )[:2]
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

    def consume_evidence_gate(self) -> bool:
        """One-shot: arm planner Inquiry Δ before deferred stall recovery."""
        if not self.evidence_gate_request:
            return False
        self.evidence_gate_request = False
        return True

    def consume_subtask_override(self) -> str | None:
        override = self.subtask_override
        self.subtask_override = None
        # Remember what we issued. `override_vla_prompt` must protect THAT string and only that
        # string: the VLA prompt reaching it as `base_prompt` is otherwise indistinguishable from
        # the VLM's own live subtask, and protecting the latter is a measured regression (job
        # 581574 t1: gate_closed_respec=249, VLA ran on "move_to_cookies" x111 and "place cookies
        # into basket" x138, score 0.0 against the incumbent's 100.0 on the brief).
        self._active_respec = override or ""
        return override

    # --- A1/A2/A3: the α interface (Actuator Guard) -------------------------------
    # The OLD signature was `override_vla_prompt(base_prompt, **_kwargs)` and both branches
    # returned `base_prompt` unchanged — a complete no-op despite carrying `stage_idx` /
    # `stage_specs` / `stage_done` / `step` in kwargs. That left the VLA running on a single
    # composite task brief for 97% of chunks (measured: H s103 t4, 970/1000 chunks). The brief
    # encodes the FIRST pass ("open all drawers in order, remember…, then put butter into…")
    # but has no semantics for REPEAT stages (07_Open_Top_Drawer_Again), so the arm spent
    # 1884 steps on stage 07 alone and scored 0 — after completing the first 6 stages in 616
    # steps. The fix: re-assert the STAGE's primitive label (from BDDL primitive_order) on
    # every chunk, so the VLA always knows WHICH drawer/container this stage is about.
    #
    # Counters are per-episode and read by the sbatch census; they are the falsifier for
    # invariants A1 (granularity alignment) and A2 (qualifier presence).
    #
    # `n_actuator_emitted_canonical` was REMOVED here (job 581642 audit): it was declared and
    # reported but never incremented by any code path, so it could only ever print 0 and the
    # census's "frac_canonical" was a constant. Repurposed as the honest count below.
    n_actuator_emitted_vlm: int = 0
    n_actuator_blocked_underspecified: int = 0
    n_qualifier_from_bddl: int = 0
    n_qualifier_from_memory: int = 0
    n_qualifier_unresolved: int = 0
    # The three exit paths of the guard, counted separately so the census can tell "the guard
    # never ran" (fatal) apart from "no aligned task stalled, so the failing path was untested"
    # (a warning).
    n_actuator_gate_open: int = 0
    n_actuator_gate_closed_healthy: int = 0
    n_actuator_gate_closed_respec: int = 0
    # A3/FAIR falsifier: how many emitted VLA prompts had an answer-key phrase masked out by the
    # guard. This replaces `n_actuator_emitted_canonical` as the "the guard actually did
    # something to the text" counter, and it is the one that matters: the guard's real job is to
    # keep the answer HM withholds away from the VLA.
    n_actuator_redacted: int = 0
    # The last subtask override THIS controller issued, as the string it was issued as.
    _active_respec: str = ""

    def override_vla_prompt(
        self,
        base_prompt: str,
        *,
        stage_idx: int = 0,
        stage_specs: list[Any] | None = None,
        stage_done: dict[str, bool] | None = None,
        step: int = 0,
        **_extra: Any,
    ) -> str:
        """α interface: protect recovery overrides; otherwise leave the planner channel alone.

        Job 582641 (local_align vs local_hm, 26×1, identical infra) falsified the old contract
        that "healthy path = restore the brief". On t22:

          HM  t=50-250  prompt="pick tomato"   → stages 01/02/03 all Y, score 100
          PG  t=80      override="pick tomato"  (one chunk)
              t=90-2500 prompt=<whole-task sentence> → 0/3 stages, score 0

        The old exit (2) rewrote EVERY non-respec chunk back to the brief, pinning the VLA on
        the whole-task sentence for 99% of chunks across all 26 tasks. HM lets the planner's
        stage primitive through. That single rewrite explained the entire −11..−22pp gap; the
        19 tasks where PG already beat HM averaged +9.0pp once the seven pinned collapses
        were set aside.

        So the guard has exactly two exits:
          1. protect a re-specification THIS controller issued (the language recovery rung);
          2. otherwise IDENTITY — pass `base_prompt` through (optionally VLA-side withhold).

        Never invent a prompt. Never force the brief. Append is still forbidden.
        """
        if not getattr(self.config, "actuator_guard", False):
            self.n_actuator_emitted_vlm += 1
            return base_prompt

        # (1) Protect the Harness's own recovery override — identity against the issued string,
        # not against "anything that isn't the brief" (that mistake was job 581574).
        respec = str(getattr(self, "_active_respec", "") or "").strip()
        if respec and base_prompt and base_prompt.strip() == respec:
            self.n_actuator_emitted_vlm += 1
            self.n_actuator_gate_closed_respec += 1
            return self._maybe_withhold(base_prompt)

        # (2) Healthy path: IDENTITY. The planner's stage primitive must reach the VLA, exactly
        # as HM does. Forcing the brief here was the 582641 collapse.
        self.n_actuator_emitted_vlm += 1
        self.n_actuator_gate_closed_healthy += 1
        return self._maybe_withhold(base_prompt)

    def _maybe_withhold(self, text: str) -> str:
        """Optionally mask answer-key phrases on the VLA prompt.

        Gated by `actuator_vla_withhold`. When False (PMH-U R1), returns `text` unchanged so
        the frozen VLA keeps destination addresses — matching HM's VLA channel. Planner-side
        redaction is independent and stays on via `PMH_REDACT_STAGE`.
        """
        if not getattr(self.config, "actuator_vla_withhold", True):
            return text
        return self._withhold(text)

    def _withhold(self, text: str) -> str:
        """Mask the task's answer-key phrases, exactly as the HM control does.

        Returns `text` unchanged when redaction is off, so non-HM arms keep their measured
        behaviour byte for byte.
        """
        if not redact.is_enabled():
            return text
        # `primitive_labels` carries the answer key for every arm (the stage names spell the
        # destination out: "place butter into top drawer"), which is the same source
        # `register_redact_answer_key` uses — so the two term sets are identical by construction.
        terms = redact.answer_key_terms(
            " ".join(str(x) for x in (getattr(self.task_info, "primitive_labels", None) or [])),
            str(getattr(self.task_info, "task_name", "") or ""),
        )
        masked = redact.mask(text, terms)
        if masked != text:
            self.n_actuator_redacted += 1
        return masked

    def actuator_guard_stats(self) -> dict[str, int]:
        """Falsifier counters for A1/A2/A3, read by the sbatch census."""
        return {
            "n_actuator_emitted_vlm": self.n_actuator_emitted_vlm,
            "n_actuator_redacted": self.n_actuator_redacted,
            "n_actuator_blocked_underspecified": self.n_actuator_blocked_underspecified,
            "n_qualifier_from_bddl": self.n_qualifier_from_bddl,
            "n_qualifier_from_memory": self.n_qualifier_from_memory,
            "n_qualifier_unresolved": self.n_qualifier_unresolved,
            "n_actuator_gate_open": self.n_actuator_gate_open,
            "n_actuator_gate_closed_healthy": self.n_actuator_gate_closed_healthy,
            "n_actuator_gate_closed_respec": self.n_actuator_gate_closed_respec,
        }

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
            **self.actuator_guard_stats(),
        }
        if self._memory_root is not None:
            self.memory.save(self._memory_root / f"task{self.task_id}" / "harness_memory.json")
        if run_dir is not None:
            self.memory.save(run_dir / "harness_memory.json")
            # Durable A1/A2/A3 census for arms without PMH (H'/HM'). The PG arm also
            # writes these via `_pmh_pact_extra` into pmh_episode_stats.json; both paths
            # are read by the sbatch census so a missing PMH dump cannot hide an inert guard.
            try:
                import json as _json
                (run_dir / "actuator_guard_stats.json").write_text(
                    _json.dumps(self.actuator_guard_stats(), indent=2), encoding="utf-8"
                )
            except Exception:
                pass

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
