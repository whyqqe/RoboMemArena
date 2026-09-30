"""ECHO zero-rewrite runtime binding: planner memory, VLA observer, and audit.

The hook is installed by a lazy import finder because `harness` is not importable at
sitecustomize time in a clean Slurm environment. No shared harness file is edited.
"""
from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from .core import EchoState, phase_hint, planner_constraint, primitive_reason, sanitize_output

_TARGETS = {"harness.api_vlm_planner", "harness.controller"}
_LOCK = threading.RLock()
_LOCAL = threading.local()
_BOUND: set[str] = set()
_INSTALLED = False
_ERRORS: list[str] = []
_N_EPISODES = 0
_ACTIVE: EchoState | None = None


def enabled() -> bool:
    return os.environ.get("MEMEXP_ECHO", "").lower() in {"1", "true", "yes"}


def _state(planner: Any = None) -> EchoState:
    global _N_EPISODES, _ACTIVE
    with _LOCK:
        if _ACTIVE is None:
            _N_EPISODES += 1
            _ACTIVE = EchoState(task_id=int(getattr(getattr(planner, "task_info", None), "task_id", 0) or 0),
                                episode=_N_EPISODES)
        if planner is not None and not _ACTIVE.task_id:
            _ACTIVE.task_id = int(getattr(getattr(planner, "task_info", None), "task_id", 0) or 0)
        return _ACTIVE


def _report() -> None:
    path = os.environ.get("MEMEXP_ECHO_REPORT", "").strip()
    if not path:
        return
    try:
        with _LOCK:
            data = {"enabled": enabled(), "bound": sorted(_BOUND), "errors": list(_ERRORS),
                    "pid": os.getpid(), "episode": _ACTIVE.snapshot() if _ACTIVE else None}
            Path(f"{path}.{os.getpid()}").write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        _ERRORS.append(f"report: {exc!r}")


def _planner(module: Any) -> None:
    cls = getattr(module, "ApiMemoryPlanner", None)
    orig = getattr(cls, "_build_messages", None)
    if orig is None or getattr(orig, "_echo_bound", False):
        return

    def build(self, *args, **kwargs):
        messages = orig(self, *args, **kwargs)
        if not enabled():
            return messages
        try:
            state = _state(self)
            state.n_plans += 1
            state.step = int(getattr(self, "step", 0) or 0)
            store = getattr(self, "frame_store_main", {}) or {}
            # Planner owns the actual image store. Write proactively while frames are
            # available, rather than relying on a controller thread-local context.
            stage = str(getattr(self, "_current_subtask", "") or "")
            if not stage:
                stage = str(getattr(getattr(self, "task_info", None), "task_block", "") or "")[:100]
            old_stage = state.stage
            state.anticipate(stage, state.step, store)
            stall = int(getattr(self, "_consecutive_same_subtask", 0) or 0)
            need = bool(old_stage and stage != old_stage) or stall >= 3
            text, indices = state.decision_package(store) if need else ("", [])
            if isinstance(messages, list):
                messages.append({"type": "text", "text": planner_constraint(state.phase)})
                state.n_constraints += 1
            if text and isinstance(messages, list):
                messages.append({"type": "text", "text": text})
                wrist = getattr(self, "frame_store_wrist", {}) or {}
                for index in indices:
                    messages.append({"type": "text", "text": f"[historical observation t={index}]"})
                    messages.append({"type": "image", "image": store[index]})
                    if getattr(self, "use_wrist", False) and index in wrist and wrist[index] is not None:
                        messages.append({"type": "image", "image": wrist[index]})
            _report()  # flush from a guaranteed-live planning seam
        except Exception as exc:
            _ERRORS.append(f"build: {exc!r}")
            _report()
        return messages
    build._echo_bound = True
    cls._build_messages = build

    api = getattr(module, "infer_primitive_via_api", None)
    if api is not None and not getattr(api, "_echo_bound", False):
        def guarded_api(*, system_prompt, user_content, **kwargs):
            if not enabled() or not isinstance(user_content, list) or not any(
                "[ECHO current physical obligation" in str(msg.get("text", ""))
                for msg in user_content if isinstance(msg, dict)
            ):
                return api(system_prompt=system_prompt, user_content=user_content, **kwargs)
            state = _state()
            phase = state.phase
            raw = api(system_prompt=system_prompt, user_content=user_content, **kwargs)
            try:
                import json as _json
                parsed = _json.loads(raw) if raw else {}
                prim = parsed.get("current_primitive", "") if isinstance(parsed, dict) else ""
                reason = primitive_reason(prim, phase) if isinstance(prim, str) else None
                if reason:
                    state.n_retries += 1
                    retry = list(user_content) + [{"type": "text", "text":
                        f"Your proposed primitive was rejected ({reason}). "
                        "Output a short physical command for the current obligation only; "
                        "keep the same JSON response schema."}]
                    revised = api(system_prompt=system_prompt, user_content=retry, **kwargs)
                    if revised:
                        raw = revised
                result, rejected = sanitize_output(raw, phase)
                if rejected:
                    state.n_rejected += 1
                _report()
                return result
            except Exception as exc:
                _ERRORS.append(f"api guard: {exc!r}")
                return raw
        guarded_api._echo_bound = True
        module.infer_primitive_via_api = guarded_api

    reset = getattr(cls, "reset_episode", None)
    if reset is not None and not getattr(reset, "_echo_bound", False):
        def reset_episode(self, *args, **kwargs):
            global _ACTIVE
            result = reset(self, *args, **kwargs)
            if enabled():
                _report()
                with _LOCK:
                    _ACTIVE = None
                _state(self)
                _report()
            return result
        reset_episode._echo_bound = True
        cls.reset_episode = reset_episode


def _controller(module: Any) -> None:
    cls = getattr(module, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None)
    if orig is not None and not getattr(orig, "_echo_bound", False):
        def observe(self, base_prompt, *args, **kwargs):
            result = orig(self, base_prompt, *args, **kwargs)
            if enabled():
                try:
                    state = _state()
                    stage_idx = int(kwargs.get("stage_idx", 0))
                    specs = kwargs.get("stage_specs") or []
                    step = int(kwargs.get("step", 0))
                    stage = str(getattr(specs[stage_idx], "name", "")) if 0 <= stage_idx < len(specs) else ""
                    # Controller and planner may execute in distinct threads. A stage hint
                    # is a decision index, NOT a credit signal. Never overwrite an
                    # active planner-thread context with this controller-thread context.
                    if state.task_id == 8 and stage:
                        state.phase = phase_hint(stage)
                    state.delivered(str(result), step)
                    state.n_identity += 1  # this observer returns the underlying result verbatim
                    if step % 25 == 0:
                        _report()
                except Exception as exc:
                    _ERRORS.append(f"observe: {exc!r}")
            return result
        observe._echo_bound = True
        cls.override_vla_prompt = observe

    end = getattr(cls, "on_episode_end", None)
    if end is not None and not getattr(end, "_echo_bound", False):
        def on_end(self, *args, **kwargs):
            if enabled():
                # Scorer values are logged for offline audit only, never fed back to
                # `independent_verification` or the planner in this episode.
                _report()
            return end(self, *args, **kwargs)
        on_end._echo_bound = True
        cls.on_episode_end = on_end


def _bind(name: str) -> None:
    module = sys.modules.get(name)
    if module is None:
        return
    try:
        if name.endswith(".controller"):
            _controller(module)
        else:
            _planner(module)
        _BOUND.add(name)
    except Exception as exc:
        _ERRORS.append(f"bind {name}: {exc!r}")
    _report()


class _Loader:
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped

    def create_module(self, spec: Any) -> Any:
        return self.wrapped.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.wrapped.exec_module(module)
        if enabled():
            _bind(module.__name__)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


class _Finder:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not enabled():
        return
    _INSTALLED = True
    sys.meta_path.insert(0, _Finder())
    for name in _TARGETS:
        if name in sys.modules:
            _bind(name)
    _report()  # persistent receipt even if no harness module was importable yet
