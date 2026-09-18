from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessConfig:
    enabled: bool = False
    max_episode_retries: int = 2
    stall_step_threshold: int = 150
    force_vlm_replan_on_stall: bool = True
    inject_vlm_context: bool = True
    inject_vla_hints: bool = False
    subtask_override_on_stall: bool = True
    stage_checkpoint_retry: bool = True
    release_gripper_on_retry: bool = True
    persist_memory: bool = True
    smart_retry: bool = True
    retry_skip_score_pct: float = 95.0
    retry_require_progress: bool = True
    skip_retry_task_ids: tuple[int, ...] = ()
    global_rules_path: Path | None = None
    api_planner_enable: bool = False
    api_key_file: Path | None = None
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = "gpt-4o-mini"
    # CGMH Mech-E: on stall, inquire once before subtask override / recovery.
    evidence_gate: bool = False
    evidence_gate_delay_steps: int = 20
    # Delay recovery without arming Inquiry (control for delay confound).
    evidence_gate_blind: bool = False
    # A1/A2: α interface — re-assert the stage's primitive label on every VLA chunk.
    # Off by default so legacy arms are bit-identical; on for the alignment arm.
    actuator_guard: bool = False
    # PMH-U R1 (measured 2026-09-14): when True, α-guard also MASKS answer-key phrases out of
    # the VLA prompt. That is the asymmetry that produced the entire HM→PG −18.8pp on the four
    # tasks where redaction registers anything (t4/t5/t11/t14): HM withholds from the PLANNER
    # only, so the frozen VLA still receives the destination address it needs to act; PG's
    # α-guard withheld from BOTH, so the VLA was handed a prompt with no address. Default True
    # keeps api_align bit-identical; api_pmh_u sets this False (planner redaction stays on).
    actuator_vla_withhold: bool = True
    # A5: RECOVERY IS A LADDER, NOT A LATCH (HarnessVLA-informed).
    #
    # Measured defect (job 582251, 26x1, PG): the stall gate latched `stall_triggered=True` on
    # first fire and only cleared it on STAGE ADVANCE. On t22 the robot spent all 2500 steps in
    # stage 1, so it received exactly ONE recovery offer (~step 80) and then nothing for the
    # remaining ~2400 steps: 250 chunks, 14 identical planner outputs ("pick tomato"),
    # stall_count 1->14, zero stages completed. t5 and t22 both ended at 0.0 while HM scored
    # 25/100 -- a physical collapse, with `failure_reason=incomplete_stage` on every task and
    # ZERO memory-attributable failures.
    #
    # The latch is worse than useless on those tasks because the language rung is a NO-OP there:
    # `_handle_stall` only emits an override when `candidate != subtask`, and on t22 the planner
    # already named the candidate ("pick tomato"), so the rung had nothing to change and still
    # burned the single latch. HarnessVLA measured the same shape across suites: a SOFT rung
    # ("remind the planner") scores ~12% while a HARD gate that changes execution scores ~32%.
    #
    # `stall_rolling` replaces the latch with a bounded, repeatable escalation. It is OFF by
    # default so the incumbent H/HM arms stay bit-identical; only the arm under test enables it.
    stall_rolling: bool = False
    # Bound on escalations per stage. Unbounded repetition of a rung that cannot help is just a
    # timer; this is the stop condition (A4's lesson, job 581446: 3.0 rewinds/ep, zero variance).
    stall_max_recoveries: int = 3
    # Consecutive no-override rungs required before the PHYSICAL rung is armed. See the
    # over-escalation note in `HarnessController._after_stall_recovery`: the stall branch also
    # raises `force_replan`, so `subtask_override is None` does NOT mean the language rung did
    # nothing. 2 = two full stall windows (~160 steps ~= 16 VLA chunks) of no language change.
    stall_noop_before_hard: int = 2

    @property
    def total_attempts(self) -> int:
        return 1 + max(0, self.max_episode_retries)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_list(raw: str | None) -> tuple[int, ...]:
    if not raw or not raw.strip():
        return ()
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, list):
        return ()
    out: list[int] = []
    for item in data:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def load_harness_config() -> HarnessConfig:
    rules_raw = os.environ.get("HARNESS_GLOBAL_RULES", "").strip()
    rules_path = Path(rules_raw) if rules_raw else None
    key_raw = os.environ.get("HARNESS_API_KEY_FILE", "").strip()
    if key_raw:
        key_path: Path | None = Path(key_raw)
    else:
        default_key = Path(__file__).resolve().parents[2] / "api_key.txt"
        key_path = default_key if default_key.is_file() else None
    return HarnessConfig(
        enabled=_truthy(os.environ.get("HARNESS_ENABLE"), default=False),
        max_episode_retries=int(os.environ.get("HARNESS_MAX_RETRIES", "2")),
        stall_step_threshold=int(os.environ.get("HARNESS_STALL_STEPS", "120")),
        force_vlm_replan_on_stall=_truthy(os.environ.get("HARNESS_FORCE_VLM_REPLAN"), default=True),
        inject_vlm_context=_truthy(os.environ.get("HARNESS_VLM_CONTEXT"), default=True),
        inject_vla_hints=_truthy(os.environ.get("HARNESS_VLA_HINTS"), default=False),
        subtask_override_on_stall=_truthy(os.environ.get("HARNESS_SUBTASK_OVERRIDE"), default=True),
        stage_checkpoint_retry=_truthy(os.environ.get("HARNESS_STAGE_CHECKPOINT"), default=True),
        release_gripper_on_retry=_truthy(os.environ.get("HARNESS_RELEASE_ON_RETRY"), default=True),
        persist_memory=_truthy(os.environ.get("HARNESS_PERSIST_MEMORY"), default=True),
        smart_retry=_truthy(os.environ.get("HARNESS_SMART_RETRY"), default=True),
        retry_skip_score_pct=float(os.environ.get("HARNESS_RETRY_SKIP_SCORE", "95")),
        retry_require_progress=_truthy(os.environ.get("HARNESS_RETRY_REQUIRE_PROGRESS"), default=True),
        skip_retry_task_ids=_parse_int_list(os.environ.get("HARNESS_SKIP_RETRY_TASKS")),
        global_rules_path=rules_path,
        api_planner_enable=_truthy(os.environ.get("HARNESS_API_PLANNER"), default=False),
        api_key_file=key_path,
        api_base_url=os.environ.get("HARNESS_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_model=os.environ.get("HARNESS_API_MODEL", "gpt-4o-mini"),
        evidence_gate=_truthy(os.environ.get("CGMH_EVIDENCE_GATE"), default=False),
        evidence_gate_delay_steps=int(os.environ.get("CGMH_GATE_DELAY_STEPS", "20")),
        evidence_gate_blind=_truthy(os.environ.get("CGMH_GATE_BLIND"), default=False),
        actuator_guard=_truthy(os.environ.get("HARNESS_ACTUATOR_GUARD"), default=False),
        # Default True: legacy api_align behaviour. Explicit "0"/"false" turns the VLA-side
        # mask off (PMH-U R1). Absent env keeps True so older arms do not silently change.
        actuator_vla_withhold=_truthy(os.environ.get("HARNESS_ACTUATOR_VLA_WITHHOLD"), default=True),
        stall_rolling=_truthy(os.environ.get("HARNESS_STALL_ROLLING"), default=False),
        stall_max_recoveries=max(1, int(os.environ.get("HARNESS_STALL_MAX_RECOVERIES", "3"))),
        stall_noop_before_hard=max(1, int(os.environ.get("HARNESS_STALL_NOOP_BEFORE_HARD", "2"))),
    )


def load_global_rules(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return _default_global_rules()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_global_rules()
    if isinstance(raw, dict) and isinstance(raw.get("rules"), list):
        return [str(x) for x in raw["rules"]]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return _default_global_rules()


def _default_global_rules() -> list[str]:
    return [
        "If grasp fails, re-localize the target before retrying contact.",
        "Do not advance to the next subtask until the current stage predicate is satisfied.",
        "Use historical keyframes to recall objects that are no longer visible.",
        "After partial progress, resume from the earliest incomplete stage instead of restarting the whole task.",
    ]
