"""EvMem-GPM binding: fluent default, gated evidence, C-controller on outputs."""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

import memexp_evmem as E

_MOD = "harness.api_vlm_planner"
_LOCAL = threading.local()
_STATE: dict[str, Any] = {
    "installed_at": None,
    "module_patched": False,
    "steps": 0,
    "steps_fluent": 0,
    "steps_evidence": 0,
    "steps_with_evidence_imgs": 0,
    "tool_calls": 0,
    "auto_looks": 0,
    "primitive_returned": 0,
    "control_rejects": 0,
    "label_rejects": 0,
    "loop_errors": 0,
    "cap_hits": 0,
    "kf_merged": 0,
    "verified_stages": [],
    "active_stage": "",
    "stage_names": [],
    "errors": [],
}
_LEDGERS: list[E.Ledger] = []
_LOCK = threading.RLock()


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def evmem_on() -> bool:
    return _truthy(os.environ.get("MEMEXP_EVMEM"))


class _Ctx:
    def __init__(self, planner, ledger: E.Ledger) -> None:
        self.planner = planner
        self.ledger = ledger
        self.msgs: list | None = None
        self.on_context: set[int] = set()
        self.step = 0
        self.recent_start = 0
        self.n_context = 0
        self.closed = False
        self.seeded = False
        self.mode = "fluent"
        self.round_offered: list[int] = []
        self.stall = 0


def _live_ctx() -> _Ctx | None:
    ctx = getattr(_LOCAL, "ctx", None)
    if ctx is None or getattr(ctx, "closed", False):
        return None
    return ctx


def _new_ctx(planner) -> _Ctx:
    prev = getattr(_LOCAL, "ctx", None)
    if prev is not None:
        prev.closed = True
    led = E.Ledger()
    with _LOCK:
        _LEDGERS.append(led)
    ctx = _Ctx(planner, led)
    _LOCAL.ctx = ctx
    _LOCAL.planner = planner
    return ctx


def _get_ctx(planner) -> _Ctx:
    ctx = _live_ctx()
    if ctx is None or getattr(_LOCAL, "planner", None) is not planner:
        return _new_ctx(planner)
    return ctx


def _seed_stages(planner, led: E.Ledger) -> None:
    try:
        task_info = getattr(planner, "task_info", None)
        task_id = int(getattr(task_info, "task_id", 0) or 0)
        if not task_id:
            return
        import importlib
        stage_mod = importlib.import_module("task2_26_reference_stage")
        names = [str(getattr(s, "name", "") or "") for s in stage_mod._task_specs(task_id)]
        led.seed_conditions(names)
        _STATE["stage_names"] = [n for n in names if n]
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"seed: {exc!r}")


def _read_stall(planner) -> int:
    try:
        return int(getattr(planner, "_consecutive_same_subtask", 0) or 0)
    except Exception:
        return 0


def _append_evidence_images(msgs: list, planner, tagged: list[tuple[int, str]], on_context: set[int]) -> int:
    if not tagged:
        return 0
    store = getattr(planner, "frame_store_main", {}) or {}
    wstore = getattr(planner, "frame_store_wrist", {}) or {}
    use_wrist = bool(getattr(planner, "use_wrist", False))
    n = 0
    for idx, tag in tagged:
        ii = int(idx)
        img = store.get(ii)
        if img is None:
            continue
        msgs.append({"type": "text", "text": f"[evmem {tag} frame abs={ii}]"})
        msgs.append({"type": "image", "image": img})
        if use_wrist:
            wrist = wstore.get(ii)
            if wrist is not None:
                msgs.append({"type": "image", "image": wrist})
        on_context.add(ii)
        n += 1
        if n >= E.EVIDENCE_CAP:
            break
    return n


def _run_auto_looks(ctx: _Ctx, store: dict) -> None:
    led = ctx.ledger
    pins = led.consume_pending_pins()
    targets = list(pins)
    if ctx.mode == "evidence":
        cur = led.active_claim()
        if cur is not None and cur.cid not in targets and cur.status != "VERIFIED":
            targets.append(cur.cid)
    for cid in targets:
        text, frames = led.look(cid, ctx.on_context, store, auto=True)
        led.record_tool(step=ctx.step, tool="look", args={"id": cid}, offered=frames,
                        text_head=text, auto=True)
        _STATE["auto_looks"] = int(_STATE["auto_looks"]) + 1
        if frames:
            ctx.round_offered.extend(int(x) for x in frames)
    led.clear_force_evidence()


def _wrap_harness_stage() -> None:
    try:
        from harness import controller as _ctl
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"stage hook import: {exc!r}")
        return
    cls = getattr(_ctl, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None) if cls is not None else None
    if orig is None or getattr(orig, "_evmem_stage_wrapped", False):
        return

    def wrapper(self, base_prompt, *, stage_idx=0, stage_specs=None, stage_done=None,
                step=0, **extra):
        try:
            names = [getattr(s, "name", "") for s in (stage_specs or [])]
            i = int(stage_idx)
            if 0 <= i < len(names) and names[i]:
                _STATE["active_stage"] = str(names[i])
            if isinstance(stage_done, dict):
                done = {str(k) for k, v in stage_done.items() if v and str(k).strip()}
                if done:
                    _STATE["verified_stages"] = sorted(set(_STATE.get("verified_stages") or []) | done)
        except Exception:
            pass
        return orig(self, base_prompt, stage_idx=stage_idx, stage_specs=stage_specs,
                    stage_done=stage_done, step=step, **extra)

    wrapper._evmem_stage_wrapped = True
    cls.override_vla_prompt = wrapper


def _wrap_build_messages(module) -> None:
    cls = module.ApiMemoryPlanner
    orig = cls._build_messages
    if getattr(orig, "_evmem_wrapped", False):
        return

    def _build_messages(self, memory_main_frames, memory_wrist_frames, context_main_frames,
                        context_wrist_frames, *, extra_memory_text=""):
        msgs = orig(self, memory_main_frames, memory_wrist_frames,
                    context_main_frames, context_wrist_frames,
                    extra_memory_text=extra_memory_text)
        if not evmem_on():
            return msgs
        try:
            ctx = _get_ctx(self)
            if not ctx.seeded:
                _seed_stages(self, ctx.ledger)
                ctx.seeded = True
            n_ctx = len(context_main_frames or [])
            recent_start = max(0, int(self.step) - n_ctx)
            on_context = set(range(recent_start, recent_start + n_ctx))
            ctx.on_context = on_context
            ctx.step = int(self.step)
            ctx.recent_start = recent_start
            ctx.n_context = n_ctx
            ctx.round_offered = []
            step_of_prev = max(0, int(self.step) - 1)
            prev = str(getattr(self, "_current_subtask", "") or "").strip()
            if prev:
                ctx.ledger.note_action(prev, step_of_prev, step_of_prev, step_of_prev + E.LOOKAHEAD)
            stage_now = str(_STATE.get("active_stage") or "").strip()
            if stage_now:
                ctx.ledger.set_active_stage(stage_now, step_of_prev)
            for nm in list(_STATE.get("verified_stages") or []):
                ctx.ledger.note_verified(nm, step_of_prev)
            try:
                if hasattr(self, "register_redact_answer_key"):
                    self.register_redact_answer_key()
            except Exception:
                pass

            stall = _read_stall(self)
            ctx.stall = stall
            mode = ctx.ledger.decide_mode(stall)
            ctx.mode = mode
            store = getattr(self, "frame_store_main", {}) or {}
            if mode == "evidence" or ctx.ledger.pending_pin_cids or ctx.ledger.force_evidence:
                ctx.mode = "evidence"
                mode = "evidence"
                _run_auto_looks(ctx, store)

            if mode == "fluent":
                ctx.ledger.n_fluent_steps += 1
                _STATE["steps_fluent"] = int(_STATE["steps_fluent"]) + 1
            else:
                ctx.ledger.n_evidence_mode_steps += 1
                _STATE["steps_evidence"] = int(_STATE["steps_evidence"]) + 1

            tagged = ctx.ledger.take_evidence_for_prompt()
            n_img = _append_evidence_images(msgs, self, tagged, on_context)
            if n_img:
                _STATE["steps_with_evidence_imgs"] = int(_STATE["steps_with_evidence_imgs"]) + 1

            redact = getattr(self, "_redact_planner_text", None)
            digest = ctx.ledger.render_digest(
                mode=mode, stall=stall,
                redact_fn=redact if callable(redact) else None,
            )
            block = digest if mode == "fluent" else (digest + "\n" + ctx.ledger.spec_text())
            msgs.append({"type": "text", "text": block})
            ctx.msgs = msgs
            _STATE["steps"] = int(_STATE["steps"]) + 1
            E.write_report(_LEDGERS)
        except Exception as exc:  # noqa: BLE001
            _STATE["loop_errors"] = int(_STATE["loop_errors"]) + 1
            _STATE["errors"].append(f"build_messages: {exc!r}")
            _LOCAL.ctx = None
        return msgs

    _build_messages._evmem_wrapped = True
    cls._build_messages = _build_messages


def _sanitize(ctx: _Ctx, out: str | None) -> str | None:
    if out is None:
        return None
    # Prefer soft rewrite via FORCE if label-like; hard rewrite via apply_control for off-graph.
    text = out
    if E.looks_like_label_primitive(text, ctx.ledger):
        ctx.ledger.n_label_rejects += 1
        _STATE["label_rejects"] = int(_STATE["label_rejects"]) + 1
    text2, rewritten = E.apply_control(text, ctx.ledger)
    if rewritten:
        _STATE["control_rejects"] = int(_STATE["control_rejects"]) + 1
        text = text2
    offered = list(ctx.round_offered) or list(ctx.ledger.evidence_abs)
    if offered:
        merged = E.merge_keyframe_positions(
            text, offered, recent_start=ctx.recent_start, n_context=ctx.n_context,
        )
        if merged != text:
            _STATE["kf_merged"] = int(_STATE["kf_merged"]) + 1
        text = merged
    return text


def _wrap_api_call(module) -> None:
    orig = module.infer_primitive_via_api
    if getattr(orig, "_evmem_wrapped", False):
        return

    def infer_primitive_via_api(*, system_prompt, user_content, **kw):
        if not evmem_on():
            return orig(system_prompt=system_prompt, user_content=user_content, **kw)
        ctx = _live_ctx()
        if ctx is None or ctx.msgs is not user_content:
            return orig(system_prompt=system_prompt, user_content=user_content, **kw)
        planner = ctx.planner
        msgs = list(user_content)
        store = getattr(planner, "frame_store_main", {}) or {}

        def _one(content):
            return orig(system_prompt=system_prompt, user_content=content, **kw)

        try:
            if ctx.mode != "evidence":
                out = _one(msgs)
                if out is None:
                    return None
                # one retry if control would reject — ask model first
                prim = E.extract_primitive(out)
                if E.control_reject_reason(prim, ctx.ledger) or E.looks_like_label_primitive(out, ctx.ledger):
                    msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
                    msgs.append({"type": "text", "text": E.FORCE_NATURAL + "\n" + "\n".join(ctx.ledger.forbidden_lines())})
                    out2 = _one(msgs)
                    if out2 is not None:
                        out = out2
                _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                return _sanitize(ctx, out)

            for round_i in range(E.MAX_ROUNDS + 1):
                out = _one(msgs)
                if out is None:
                    return None
                call = E.parse_tool_call(out, E.known_tools())
                if call is None:
                    prim = E.extract_primitive(out)
                    if E.control_reject_reason(prim, ctx.ledger) or E.looks_like_label_primitive(out, ctx.ledger):
                        msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
                        msgs.append({"type": "text", "text": E.FORCE_NATURAL + "\n" + "\n".join(ctx.ledger.forbidden_lines())})
                        out2 = _one(msgs)
                        if out2 is not None and E.parse_tool_call(out2, E.known_tools()) is None:
                            out = out2
                    _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                    return _sanitize(ctx, out)
                if round_i >= E.MAX_ROUNDS:
                    _STATE["cap_hits"] = int(_STATE["cap_hits"]) + 1
                    msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
                    msgs.append({"type": "text", "text": E.FORCE_FINAL})
                    forced = _one(msgs)
                    _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                    return _sanitize(ctx, forced if forced is not None else out)
                text, frames = E.dispatch(
                    ctx.ledger, call["tool"], dict(call.get("args") or {}),
                    on_context=ctx.on_context, frame_store=store,
                )
                ctx.ledger.record_tool(step=ctx.step, tool=call["tool"],
                                       args=dict(call.get("args") or {}),
                                       offered=frames, text_head=text, auto=False)
                if frames:
                    ctx.round_offered.extend(int(x) for x in frames)
                _STATE["tool_calls"] = int(_STATE["tool_calls"]) + 1
                E.write_report(_LEDGERS)
                msgs.append({"type": "text", "text": f"Your previous output was:\n{out}"})
                msgs.append({"type": "text", "text": text})
                for idx in frames:
                    img = store.get(int(idx))
                    if img is not None:
                        msgs.append({"type": "image", "image": img})
                        wrist = getattr(planner, "frame_store_wrist", {}).get(int(idx))
                        if wrist is not None and getattr(planner, "use_wrist", False):
                            msgs.append({"type": "image", "image": wrist})
                    ctx.on_context.add(int(idx))
            return _sanitize(ctx, _one(msgs))
        except Exception as exc:  # noqa: BLE001
            _STATE["loop_errors"] = int(_STATE["loop_errors"]) + 1
            _STATE["errors"].append(f"tool loop: {exc!r}")
            try:
                return orig(system_prompt=system_prompt, user_content=user_content, **kw)
            except Exception:
                return None

    infer_primitive_via_api._evmem_wrapped = True
    module.infer_primitive_via_api = infer_primitive_via_api


def _wrap_resets(module) -> None:
    cls = module.ApiMemoryPlanner
    for name in ("reset_episode", "set_task_info"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_evmem_wrapped", False):
            continue

        def make(orig_fn):
            def wrapper(self, *a, **k):
                out = orig_fn(self, *a, **k)
                if evmem_on():
                    ctx = getattr(_LOCAL, "ctx", None)
                    if ctx is not None:
                        ctx.closed = True
                    _LOCAL.ctx = None
                    _STATE["verified_stages"] = []
                    _STATE["active_stage"] = ""
                return out
            wrapper._evmem_wrapped = True
            return wrapper

        setattr(cls, name, make(orig))


def _bind(module) -> None:
    try:
        _wrap_build_messages(module)
        _wrap_api_call(module)
        _wrap_resets(module)
        _wrap_harness_stage()
        _STATE["module_patched"] = True
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"bind failed: {exc!r}")
        sys.stderr.write(f"[evmem] BIND FAILED: {exc!r}\n")


def install() -> None:
    if _STATE["installed_at"] is not None:
        return
    _STATE["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    import atexit
    atexit.register(lambda: E.write_report(_LEDGERS))

    from importlib.abc import MetaPathFinder
    from importlib.machinery import PathFinder

    class _PostExecLoader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            return self._wrapped.create_module(spec)

        def exec_module(self, module):
            self._wrapped.exec_module(module)
            if evmem_on():
                _bind(module)

        def __getattr__(self, item):
            return getattr(self._wrapped, item)

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _MOD:
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PostExecLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
    module = sys.modules.get(_MOD)
    if module is not None and evmem_on():
        _bind(module)
