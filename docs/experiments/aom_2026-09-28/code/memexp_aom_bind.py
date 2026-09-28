"""AOM binding: one obligation ledger, one arbitration law, one board.

This mirrors `memexp_evmem_bind` deliberately, down to the hook points and the report cadence, so
that an AOM-vs-GPM difference is attributable to the LEDGER AND THE LAW and not to a different way
of reaching into `harness.*`. The one structural difference is that GPM has two renderers (a digest
and, in evidence mode, a spec) and AOM has ONE board, generated from (mode, status).

Floor F lives here in its operational form: an AOM auto-look runs whenever a GPM auto-look would,
which is whenever the verdict is `retrieve` OR `stagnant`. `stagnant` ADDS a sentence to the board;
it does not remove a retrieval. That is the invariant PIC-MEM v3 violated, and it is the reason its
evidence cadence collapsed from 8-15 frames/episode to 4.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

import memexp_aom as A

_MOD = "harness.api_vlm_planner"
_LOCAL = threading.local()
_STATE: dict[str, Any] = {
    "installed_at": None,
    "module_patched": False,
    "steps": 0,
    "steps_fluent": 0,
    "steps_evidence": 0,
    "steps_act": 0,
    "steps_retrieve": 0,
    "steps_derive": 0,
    "steps_stagnant": 0,
    "steps_advance": 0,
    "auto_looks": 0,
    "gate_rejects": 0,
    "offgraph_rejects": 0,
    "dep_rejects": 0,
    "label_rejects": 0,
    "derived_recomputes": 0,
    "primitive_returned": 0,
    "tool_calls": 0,
    "cap_hits": 0,
    "loop_errors": 0,
    "active_stage": "",
    "verified_stages": [],
    "stage_names": [],
    "exec_labels": [],
    "name_objects": False,
    "errors": [],
}
_LEDGERS: list[A.Ledger] = []
_LOCK = threading.RLock()


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def aom_on() -> bool:
    return _truthy(os.environ.get("MEMEXP_AOM"))


class _Ctx:
    def __init__(self, planner, ledger: A.Ledger) -> None:
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
        self.verdict = "advance"
        self.round_offered: list[int] = []


def _live_ctx() -> _Ctx | None:
    ctx = getattr(_LOCAL, "ctx", None)
    if ctx is None or getattr(ctx, "closed", False):
        return None
    return ctx


def _new_ctx(planner) -> _Ctx:
    prev = getattr(_LOCAL, "ctx", None)
    if prev is not None:
        prev.closed = True
    led = A.Ledger()
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


def _seed_obligations(planner, led: A.Ledger) -> None:
    """Seed from the EXECUTION partition, not the scoring partition.

    The defect this replaces: `_task_specs(task_id)` is the SCORING partition, and on task 19 it
    lists only the three `Place_*_Cabinet2` stages, so the graph contained no grasp obligation at
    all -- and the board then commanded a place (unsatisfiable without a held object) for the
    whole episode. `task_info.primitive_labels` is the BDDL EXECUTION partition, and
    `harness.stage_mapper.expected_primitive_for_stage` is the harness's OWN bridge between the
    two (it exists for the stall ladder and the ledger guard). AOM uses it rather than inventing a
    second mapping.
    """
    try:
        task_info = getattr(planner, "task_info", None)
        task_id = int(getattr(task_info, "task_id", 0) or 0)
        if not task_id:
            return
        import importlib
        stage_mod = importlib.import_module("task2_26_reference_stage")
        specs = list(stage_mod._task_specs(task_id) or [])
        scored = [str(getattr(s, "name", "") or "") for s in specs]
        labels = [str(x) for x in (getattr(task_info, "primitive_labels", None) or [])]

        if labels and specs:
            mode = str(os.environ.get("MEMEXP_AOM_EXEC_SEED", "auto") or "auto").strip().lower()
            use_exec, why = A.should_seed_execution(scored, labels)
            if mode in {"0", "false", "no", "off"}:
                use_exec, why = False, "disabled by MEMEXP_AOM_EXEC_SEED"
            elif mode in {"1", "true", "yes", "on"}:
                use_exec, why = True, "forced by MEMEXP_AOM_EXEC_SEED"
            _STATE["exec_seed_why"] = why
            if use_exec:
                mapping: dict[str, str] = {}
                try:
                    from harness.stage_mapper import expected_primitive_for_stage as _exp
                    for i, sp in enumerate(specs):
                        lab = _exp(primitive_labels=labels, stage_idx=i, stage_specs=specs,
                                   stage_done={s.name: False for s in specs[:i]},
                                   current_subtask="")
                        if lab and lab not in mapping:
                            mapping[lab] = scored[i] if i < len(scored) else ""
                except Exception as exc:  # noqa: BLE001
                    _STATE["errors"].append(f"stage_mapper: {exc!r}")
                # Safety: if the bridge resolved nothing, degrade to the scoring partition rather
                # than ship a graph whose stages can never settle.
                if any(v for v in mapping.values()):
                    mt = str(getattr(task_info, "memory_type", "") or "").upper()
                    name_objects = "O" not in mt
                    led.seed_execution(labels, scored, mapping, name_objects=name_objects)
                    _STATE["stage_names"] = scored
                    _STATE["exec_labels"] = labels
                    _STATE["name_objects"] = name_objects
                    return
                _STATE["errors"].append("stage_mapper: no stage bound; fell back to scoring partition")
            else:
                _STATE["errors"].append(f"exec seed skipped: {why}")
        led.seed(scored)
        _STATE["stage_names"] = scored
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"seed: {exc!r}")


def _read_stall(planner) -> int:
    try:
        return int(getattr(planner, "_consecutive_same_subtask", 0) or 0)
    except Exception:
        return 0


def _append_evidence_images(msgs: list, planner, tagged: list[tuple[int, str]],
                            on_context: set[int]) -> int:
    if not tagged:
        return 0
    store = getattr(planner, "frame_store_main", {}) or {}
    wstore = getattr(planner, "frame_store_wrist", {}) or {}
    use_wrist = bool(getattr(planner, "use_wrist", False))
    n = 0
    for idx, oid in tagged:
        ii = int(idx)
        img = store.get(ii)
        if img is None:
            continue
        msgs.append({"type": "text", "text": f"[aom {oid} frame abs={ii}]"})
        msgs.append({"type": "image", "image": img})
        if use_wrist:
            wrist = wstore.get(ii)
            if wrist is not None:
                msgs.append({"type": "image", "image": wrist})
        on_context.add(ii)
        n += 1
        if n >= A.EVIDENCE_CAP:
            break
    return n


def _run_auto_retrieval(ctx: _Ctx, store: dict, verdict: str, oid: str) -> None:
    """The floor F in operation: retrieval runs for `retrieve` AND for `stagnant`, so declaring
    stagnation never costs the evidence channel a frame."""
    led = ctx.ledger
    targets = list(led.consume_pins())
    if verdict in {"retrieve", "stagnant"} and oid and oid not in targets:
        targets.append(oid)
    for t in targets:
        text, frames = led.look(t, ctx.on_context, store, auto=True)
        led.record_tool(step=ctx.step, tool="look", args={"id": t}, offered=frames,
                        text_head=text, auto=True)
        _STATE["auto_looks"] = int(_STATE["auto_looks"]) + 1
        if frames:
            ctx.round_offered.extend(int(x) for x in frames)
        if verdict == "stagnant" and t in led.obligations:
            led.obligations[t].stagnant = True
    led.clear_force_retrieve()


def _wrap_harness_stage() -> None:
    try:
        from harness import controller as _ctl
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"stage hook import: {exc!r}")
        return
    cls = getattr(_ctl, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None) if cls is not None else None
    if orig is None or getattr(orig, "_aom_stage_wrapped", False):
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

    wrapper._aom_stage_wrapped = True
    cls.override_vla_prompt = wrapper


def _wrap_build_messages(module) -> None:
    cls = module.ApiMemoryPlanner
    orig = cls._build_messages
    if getattr(orig, "_aom_wrapped", False):
        return

    def _build_messages(self, memory_main_frames, memory_wrist_frames, context_main_frames,
                        context_wrist_frames, *, extra_memory_text=""):
        msgs = orig(self, memory_main_frames, memory_wrist_frames,
                    context_main_frames, context_wrist_frames,
                    extra_memory_text=extra_memory_text)
        if not aom_on():
            return msgs
        try:
            ctx = _get_ctx(self)
            if not ctx.seeded:
                _seed_obligations(self, ctx.ledger)
                ctx.seeded = True
            n_ctx = len(context_main_frames or [])
            recent_start = max(0, int(self.step) - n_ctx)
            ctx.on_context = set(range(recent_start, recent_start + n_ctx))
            ctx.step = int(self.step)
            ctx.recent_start = recent_start
            ctx.n_context = n_ctx
            ctx.round_offered = []
            step_of_prev = max(0, int(self.step) - 1)
            prev = str(getattr(self, "_current_subtask", "") or "").strip()
            if prev:
                ctx.ledger.note_action(prev, step_of_prev, step_of_prev, step_of_prev + A.LOOKAHEAD)
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
            state = ctx.ledger
            state.recompute_derived(step_of_prev)
            verdict, oid = state.arbitrate(stall)
            state.last_verdict = verdict
            ctx.verdict = verdict
            # `mode` selects the BOARD VARIANT, not whether a board is emitted. GPM renders its
            # digest in both modes; AOM renders its board in both modes, and `mode` only decides
            # whether the admissible-template and tool sections appear. Emitting nothing in `act`
            # mode would silently drop the forbidden set, which is half of what the gate is for.
            mode = "evidence" if verdict in {"retrieve", "stagnant"} else "fluent"
            # Floor F: the evidence channel runs for retrieve AND stagnant.
            if verdict in {"retrieve", "stagnant"} or state.pending_pins or state.force_retrieve:
                mode = "evidence"
                _run_auto_retrieval(ctx, getattr(self, "frame_store_main", {}) or {}, verdict, oid)
            # A settled ledger says nothing: behaviour is then `nomem`'s, byte for byte.
            if state.fluent_equivalent():
                mode = "fluent"
            ctx.mode = mode

            if mode == "evidence":
                state.n_evidence_mode_steps += 1
                _STATE["steps_evidence"] = int(_STATE["steps_evidence"]) + 1
            else:
                state.n_fluent += 1
                _STATE["steps_fluent"] = int(_STATE["steps_fluent"]) + 1
            if verdict == "act":
                state.n_act += 1
                _STATE["steps_act"] = int(_STATE["steps_act"]) + 1
            elif verdict == "retrieve":
                state.n_retrieve += 1
                _STATE["steps_retrieve"] = int(_STATE["steps_retrieve"]) + 1
            elif verdict == "derive":
                state.n_derive += 1
                _STATE["steps_derive"] = int(_STATE["steps_derive"]) + 1
            elif verdict == "stagnant":
                state.n_stagnant += 1
                _STATE["steps_stagnant"] = int(_STATE["steps_stagnant"]) + 1
            elif verdict == "advance":
                state.n_advance += 1
                _STATE["steps_advance"] = int(_STATE["steps_advance"]) + 1

            tagged = state.take_evidence()
            n_img = _append_evidence_images(msgs, self, tagged, ctx.on_context)

            redact = getattr(self, "_redact_planner_text", None)
            block = state.board(mode=mode, stall=stall,
                                redact_fn=redact if callable(redact) else None)
            msgs.append({"type": "text", "text": block})
            if mode == "evidence" and n_img == 0 and not tagged:
                # Only advertise the tools when a retrieval is actually possible; an empty tool
                # block is an invitation to spend a round on nothing.
                msgs.append({"type": "text", "text": state.spec_text()})
            ctx.msgs = msgs
            _STATE["steps"] = int(_STATE["steps"]) + 1
            _STATE["derived_recomputes"] = int(_STATE["derived_recomputes"]) + 1
            A.write_report(_LEDGERS)
        except Exception as exc:  # noqa: BLE001
            _STATE["loop_errors"] = int(_STATE["loop_errors"]) + 1
            _STATE["errors"].append(f"build_messages: {exc!r}")
            _LOCAL.ctx = None
        return msgs

    _build_messages._aom_wrapped = True
    cls._build_messages = _build_messages


def _sanitize(ctx: _Ctx, out: str | None) -> str | None:
    if out is None:
        return None
    text = out
    if A.G.looks_like_label_primitive(text, None):
        ctx.ledger.n_label_rejects += 1
        _STATE["label_rejects"] = int(_STATE["label_rejects"]) + 1
    text2, rewritten = ctx.ledger.apply_control(text)
    if rewritten:
        _STATE["gate_rejects"] = int(_STATE["gate_rejects"]) + 1
        text = text2
    offered = list(ctx.round_offered) or list(ctx.ledger.evidence_abs)
    if offered:
        text = A.merge_keyframe_positions(
            text, offered, recent_start=ctx.recent_start, n_context=ctx.n_context,
        )
    return text


def _wrap_api_call(module) -> None:
    orig = module.infer_primitive_via_api
    if getattr(orig, "_aom_wrapped", False):
        return

    def infer_primitive_via_api(*, system_prompt, user_content, **kw):
        if not aom_on():
            return orig(system_prompt=system_prompt, user_content=user_content, **kw)
        ctx = _live_ctx()
        if ctx is None or ctx.msgs is not user_content:
            return orig(system_prompt=system_prompt, user_content=user_content, **kw)
        planner = ctx.planner
        msgs = list(user_content)
        store = getattr(planner, "frame_store_main", {}) or {}

        def _one(content):
            return orig(system_prompt=system_prompt, user_content=content, **kw)

        def _rejects(out):
            prim = A.G.extract_primitive(out)
            return ctx.ledger.admissibility(prim) or A.G.looks_like_label_primitive(out, None)

        def _ask_again(msgs, out):
            msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
            msgs.append({"type": "text",
                         "text": A.FORCE_NATURAL + "\n" + "\n".join(ctx.ledger.forbidden_lines())})
            out2 = _one(msgs)
            return out2 if out2 is not None else out

        try:
            for round_i in range(A.MAX_ROUNDS + 1):
                out = _one(msgs)
                if out is None:
                    return None
                call = A.parse_tool_call(out, ctx.ledger.known_tools())
                if call is None:
                    if _rejects(out):
                        out = _ask_again(msgs, out)
                    _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                    return _sanitize(ctx, out)
                if round_i >= A.MAX_ROUNDS:
                    _STATE["cap_hits"] = int(_STATE["cap_hits"]) + 1
                    msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
                    msgs.append({"type": "text", "text": A.FORCE_FINAL})
                    forced = _one(msgs)
                    _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                    return _sanitize(ctx, forced if forced is not None else out)
                text, frames = ctx.ledger.dispatch(
                    call["tool"], dict(call.get("args") or {}),
                    on_context=ctx.on_context, store=store,
                )
                ctx.ledger.record_tool(step=ctx.step, tool=call["tool"],
                                       args=dict(call.get("args") or {}),
                                       offered=frames, text_head=text, auto=False)
                if frames:
                    ctx.round_offered.extend(int(x) for x in frames)
                _STATE["tool_calls"] = int(_STATE["tool_calls"]) + 1
                A.write_report(_LEDGERS)
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

    infer_primitive_via_api._aom_wrapped = True
    module.infer_primitive_via_api = infer_primitive_via_api


def _wrap_resets(module) -> None:
    cls = module.ApiMemoryPlanner
    for name in ("reset_episode", "set_task_info"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_aom_wrapped", False):
            continue

        def make(orig_fn):
            def wrapper(self, *a, **k):
                out = orig_fn(self, *a, **k)
                if aom_on():
                    ctx = getattr(_LOCAL, "ctx", None)
                    if ctx is not None:
                        ctx.closed = True
                    _LOCAL.ctx = None
                    _STATE["verified_stages"] = []
                    _STATE["active_stage"] = ""
                return out
            wrapper._aom_wrapped = True
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
        sys.stderr.write(f"[aom] BIND FAILED: {exc!r}\n")


def install() -> None:
    if _STATE["installed_at"] is not None:
        return
    _STATE["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    import atexit

    def _final() -> None:
        _STATE["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        A.write_report(_LEDGERS)

    atexit.register(_final)

    from importlib.abc import MetaPathFinder
    from importlib.machinery import PathFinder

    class _PostExecLoader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            return self._wrapped.create_module(spec)

        def exec_module(self, module):
            self._wrapped.exec_module(module)
            if aom_on():
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
    if module is not None and aom_on():
        _bind(module)
