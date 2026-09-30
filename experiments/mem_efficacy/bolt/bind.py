"""Planner bind layer for BOLT-Sync.

Mirrors memexp_aom_bind's hook points so BOLT is a single-variable swap against AOM:
  * ApiMemoryPlanner._build_messages  — inject board + Decision Need
  * infer_primitive_via_api           — arbitrate / rewrite
  * HarnessController.override_vla_prompt — track verified stages
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from .harness import BoltHarness, enabled
from .types import AV_ACT, AV_REWRITE, ActionContract

_STATE: dict[str, Any] = {
    "installed_at": None,
    "errors": [],
    "active_stage": "",
    "verified_stages": [],
    "exec_labels": [],
    "scored_names": [],
    "name_objects": True,
    "n_episodes": 0,
    "n_build": 0,
    "n_report": 0,
    "n_arbitrate": 0,
    "verdicts": {},
    "bridge": {},
    "alignment": {},
    "unbridged_stages": [],
    "report_every": 0,
}
_LOCAL = threading.local()
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def bolt_on() -> bool:
    return enabled()


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in _TRUTHY


class _Ctx:
    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.harness = BoltHarness()
        self.seeded = False
        self.step = 0
        self.on_context: set[int] = set()
        self.last_contract: ActionContract | None = None


def _live_ctx() -> _Ctx | None:
    return getattr(_LOCAL, "ctx", None)


def _new_ctx(planner: Any) -> _Ctx:
    ctx = _Ctx(planner)
    _LOCAL.ctx = ctx
    _LOCAL.planner = planner
    _STATE["n_episodes"] = int(_STATE.get("n_episodes") or 0) + 1
    return ctx


def _get_ctx(planner: Any) -> _Ctx:
    ctx = _live_ctx()
    if ctx is None or getattr(_LOCAL, "planner", None) is not planner:
        return _new_ctx(planner)
    return ctx


def _seed(planner: Any, h: BoltHarness) -> None:
    """ALWAYS seed from the execution partition (BOLT's structural claim)."""
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
        if not labels:
            # fall back to scored names only if BDDL labels are missing (should be rare)
            labels = [s.replace("_", " ").lower() for s in scored]
            _STATE["errors"].append("no primitive_labels; fell back to scored names as exec")

        mapping: dict[str, str] = {}
        if labels and specs:
            try:
                try:
                    from .align import describe_alignment  # noqa: PLC0415
                except ImportError:  # pragma: no cover - absolute fallback
                    from bolt.align import describe_alignment  # noqa: PLC0415
                info = describe_alignment(labels, specs)
                mapping = dict(info.get("bridge") or {})
                _STATE["alignment"] = info
            except Exception as exc:  # noqa: BLE001
                _STATE["errors"].append(f"align: {exc!r}")

        mt = str(getattr(task_info, "memory_type", "") or "").upper()
        name_objects = "O" not in mt  # transfer / counting / sequence name objects
        h.seed(labels, scored, mapping, name_objects=name_objects)
        _STATE["exec_labels"] = labels
        _STATE["scored_names"] = scored
        _STATE["name_objects"] = name_objects
        _STATE["bridge"] = dict(mapping)
        # a scored stage no node can carry is a permanent score hole: surface it
        _STATE["unbridged_stages"] = [
            s for s in scored if s and s not in set(mapping.values())
        ]
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"seed: {exc!r}")
        h.errors.append(f"seed: {exc!r}")


def _wrap_harness_stage() -> None:
    try:
        from harness import controller as _ctl
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"stage hook import: {exc!r}")
        return
    cls = getattr(_ctl, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None) if cls is not None else None
    if orig is None or getattr(orig, "_bolt_stage_wrapped", False):
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
                    prev = set(_STATE.get("verified_stages") or [])
                    _STATE["verified_stages"] = sorted(prev | done)
                    ctx = _live_ctx()
                    if ctx is not None:
                        ctx.harness.note_verified_stages(done, step=int(step))
        except Exception:
            pass
        return orig(self, base_prompt, stage_idx=stage_idx, stage_specs=stage_specs,
                    stage_done=stage_done, step=step, **extra)

    wrapper._bolt_stage_wrapped = True
    cls.override_vla_prompt = wrapper


def _wrap_build_messages(module) -> None:
    cls = module.ApiMemoryPlanner
    orig = cls._build_messages
    if getattr(orig, "_bolt_wrapped", False):
        return

    def _build_messages(self, memory_main_frames, memory_wrist_frames, context_main_frames,
                        context_wrist_frames, *, extra_memory_text=""):
        msgs = orig(self, memory_main_frames, memory_wrist_frames,
                    context_main_frames, context_wrist_frames,
                    extra_memory_text=extra_memory_text)
        if not bolt_on():
            return msgs
        try:
            ctx = _get_ctx(self)
            if not ctx.seeded:
                _seed(self, ctx.harness)
                ctx.seeded = True
            ctx.step = int(getattr(self, "step", 0) or 0)
            # ingest any stages the harness has verified since last call
            verified = list(_STATE.get("verified_stages") or [])
            if verified:
                ctx.harness.note_verified_stages(verified, step=ctx.step)
            ctx.harness.observe({"verified_stages": verified})
            need = ctx.harness.compile_decision_need()
            board = ctx.harness.context(need)
            if board:
                msgs.append({"type": "text", "text": board})
            # --- telemetry -----------------------------------------------------------------
            # BOLT v1 never archived a single report: the old hook only fired on
            # `reset_episode`, which never sees a live, seeded ctx, so the Φ curve and
            # every arbiter/router counter were lost and the first failure analysis had
            # to be reconstructed from the model's echo of the board.  `_build_messages`
            # is the one guaranteed-live lifecycle point, so archive from here.
            _STATE["n_build"] = int(_STATE.get("n_build") or 0) + 1
            every = int(_STATE.get("report_every") or 0)
            if every > 0 and _STATE["n_build"] % every == 0:
                _write_report(ctx)
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"].append(f"build_messages: {exc!r}")
        return msgs

    _build_messages._bolt_wrapped = True
    cls._build_messages = _build_messages


def resolve_planning_call(kwargs: dict[str, Any]) -> tuple[Any, bool]:
    """Locate the planner and decide whether this is the *planning* call.

    Returns (planner, is_planning_call).  The planner comes from the live
    thread-local context, because `infer_primitive_via_api` is invoked with pure
    keyword arguments and carries no planner handle.  The planning call is then
    identified by its `system_prompt` being the planner's own: that same function
    is shared with the memory-access decision, the PMH decision and the stage
    visual verifier, and rewriting any of those would corrupt them (the verifier's
    output is a verdict, not a robot command).
    """
    planner = getattr(_LOCAL, "planner", None)
    if planner is None or _live_ctx() is None:
        planner = kwargs.get("planner")
    if planner is None:
        return None, False
    want = getattr(planner, "system_prompt", None)
    got = kwargs.get("system_prompt")
    if want is None or got is None or str(got) != str(want):
        return planner, False
    return planner, True


def _wrap_api_call(module) -> None:
    """Arbitrate the planner's emitted primitive; rewrite on reject.

    MEASURED DEFECT (found while re-analysing the archived v1 runs): this hook was
    INERT for every BOLT episode ever run.  `harness.api_vlm_planner` calls
    `infer_primitive_via_api(system_prompt=..., user_content=..., ...)` with pure
    keyword arguments and no planner handle -- see the planning call inside
    `ApiMemoryPlanner.infer_sync`.  The old lookup looked for a `planner` kwarg or a
    positional argument with `_build_messages`, found neither, and returned the
    response untouched.  Consequences, all of which were mis-attributed to the graph:

      * the Arbiter never ran: no rewrite, no HOLD, no claim or clock check;
      * `harness.arbitrate` never ran, so `note_attempt` never fired, so the graph
        never learned that a node had been attempted and ACTIVE could never advance.

    The fix resolves the planner from the live thread-local context (set by
    `_build_messages`, which `infer_sync` calls immediately before this one) and then
    verifies it is really the *planning* call.  That guard is not defensive padding:
    `infer_primitive_via_api` is shared with the memory-access decision, the PMH
    decision and the stage visual verifier, and arbitrating any of those would
    corrupt them (the verifier's output is a verdict, not a robot command).  The
    planning call is identified by its system prompt being the planner's own.
    """
    orig = getattr(module, "infer_primitive_via_api", None)
    if orig is None or getattr(orig, "_bolt_wrapped", False):
        return

    def _bump(verdict: str) -> None:
        _STATE["verdicts"] = _STATE.get("verdicts") or {}
        _STATE["verdicts"][verdict] = int(_STATE["verdicts"].get(verdict, 0)) + 1

    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not bolt_on():
            return out
        try:
            planner, is_planning_call = resolve_planning_call(kwargs)
            if planner is None or not is_planning_call:
                return out
            ctx = _get_ctx(planner)
            if not ctx.seeded:
                _seed(planner, ctx.harness)
                ctx.seeded = True
            text = out
            if isinstance(out, tuple):
                text = out[0]
            prim = str(text or "").strip()
            # Extract current_primitive from JSON if needed
            try:
                import memexp_evmem as G
                extracted = G.extract_primitive(prim)
                if extracted:
                    prim = extracted
            except Exception:
                pass
            decision = ctx.harness.arbitrate(prim)
            ctx.last_contract = decision.contract
            _STATE["n_arbitrate"] = int(_STATE.get("n_arbitrate") or 0) + 1
            _bump(decision.verdict)
            if decision.verdict == AV_ACT:
                return out  # accepted as-is: no need to re-serialize
            if decision.verdict == AV_REWRITE and decision.contract.primitive:
                # rejected → serve the corrected contract in place of the proposal
                if prim and prim.lstrip().startswith("{"):
                    try:
                        new_text = _repack_json(text, decision.contract.primitive)
                    except Exception:
                        new_text = decision.contract.primitive
                else:
                    new_text = decision.contract.primitive
                if isinstance(out, tuple):
                    return (new_text,) + out[1:]
                return new_text
            # hold / recall / observe / recover: leave text, annotate via board next step
            return out
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"].append(f"api_call: {exc!r}")
            return out

    wrapper._bolt_wrapped = True
    module.infer_primitive_via_api = wrapper


def _repack_json(original: str, primitive: str) -> str:
    try:
        obj = json.loads(original) if original.lstrip().startswith("{") else None
    except Exception:
        obj = None
    if isinstance(obj, dict):
        obj["current_primitive"] = primitive
        return json.dumps(obj)
    return json.dumps({"current_primitive": primitive, "keyframe_positions": []})


def _wrap_resets(module) -> None:
    """Clear per-episode state at every episode boundary.

    MEASURED DEFECT (task 8 v4): `reset_episode` cleared the thread-local ctx but
    left `_STATE['verified_stages']` intact.  The next episode's `_build_messages`
    then replayed the previous episode's scored stages into a freshly seeded
    graph, so Lift and Pour_One arrived pre-SETTLED and ACTIVE jumped straight to
    `pour_two` — which can never score without the earlier stages.  Observed
    score series `0, 66.7, 0, 0, 0` then died on consec_zero.  AOM's bind clears
    the same keys; BOLT must too.
    """
    cls = module.ApiMemoryPlanner
    for name in ("reset", "reset_episode", "begin_episode", "set_task_info"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, f"_bolt_{name}_wrapped", False):
            continue

        def make(o, n):
            def wrapped(self, *a, **k):
                try:
                    ctx = _live_ctx()
                    if ctx is not None and ctx.harness.seeded:
                        _write_report(ctx)
                except Exception:
                    pass
                _LOCAL.ctx = None
                _LOCAL.planner = None
                _STATE["verified_stages"] = []
                _STATE["active_stage"] = ""
                return o(self, *a, **k)
            setattr(wrapped, f"_bolt_{n}_wrapped", True)
            return wrapped
        setattr(cls, name, make(orig, name))


def _write_report(ctx: _Ctx) -> None:
    path = str(os.environ.get("MEMEXP_BOLT_REPORT", "")).strip()
    if not path:
        return
    try:
        report = ctx.harness.report()
        report["installed_at"] = _STATE.get("installed_at")
        report["errors"] = list(_STATE.get("errors") or []) + list(ctx.harness.errors)
        report["binding"] = {
            "n_episodes": _STATE.get("n_episodes"),
            "n_build": _STATE.get("n_build"),
            "n_report": _STATE.get("n_report"),
            "n_arbitrate": _STATE.get("n_arbitrate"),
            "arbiter_verdicts": dict(_STATE.get("verdicts") or {}),
            "exec_labels": list(_STATE.get("exec_labels") or []),
            "scored_names": list(_STATE.get("scored_names") or []),
            "bridge": dict(_STATE.get("bridge") or {}),
            "alignment": dict(_STATE.get("alignment") or {}),
            "unbridged_stages": list(_STATE.get("unbridged_stages") or []),
        }
        pid = os.getpid()
        out = f"{path}.{pid}"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        _STATE["n_report"] = int(_STATE.get("n_report") or 0) + 1
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"report: {exc!r}")


def install() -> None:
    if _STATE["installed_at"] is not None:
        return
    _STATE["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        _STATE["report_every"] = int(os.environ.get("MEMEXP_BOLT_REPORT_EVERY", "25") or 0)
    except Exception:  # noqa: BLE001
        _STATE["report_every"] = 25
    try:
        import harness.api_vlm_planner as module  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"import api_vlm_planner: {exc!r}")
        return
    try:
        _wrap_harness_stage()
        _wrap_build_messages(module)
        _wrap_api_call(module)
        _wrap_resets(module)
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"install: {exc!r}")
