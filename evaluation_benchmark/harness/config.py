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
        api_base_url=os.environ.get("HARNESS_API_BASE_URL", "https://api.openai-proxy.org/v1"),
        api_model=os.environ.get("HARNESS_API_MODEL", "gpt-4o-mini"),
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
