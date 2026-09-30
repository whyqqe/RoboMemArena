"""Planner-channel binding for DIAL.

Three measured defects in this repo's bind layers constrain this file, and all three are
avoided *by construction* rather than by care:

1.  REWRITING THE VLA PROMPT DESTROYS THE RUN.  `HarnessController.override_vla_prompt`
    documents job 582641: rewriting every non-respec chunk to the brief pinned the VLA on
    the whole-task sentence for 99% of chunks and took t22 from 100 to 0 (-11..-22pp over
    the suite).  Its contract is now "never invent a prompt, append is forbidden".
    -> DIAL wraps `override_vla_prompt` as an OBSERVER ONLY and returns `orig(...)`
       untouched.  `selftest.py::test_override_is_identity` asserts byte equality, so DIAL
       cannot incur that failure mode at all.

2.  `infer_primitive_via_api` IS A TRAP.  BOLT's bind records that its arbiter hook was
    INERT for every archived episode: the function is called with pure kwargs and carries
    no planner handle, and it is shared with the memory-access decision, the PMH decision
    and the stage visual verifier, so rewriting by mistake corrupts non-planning calls.
    -> DIAL does NOT wrap it.  There is no rewrite path anywhere in DIAL — which is also
       the architecture's point (no BLOCK) — so defect 2 cannot be reached.

3.  CROSS-EPISODE CONTAMINATION.  BOLT's bind records task 8 v4: `reset_episode` cleared
    the thread-local context but left the module-global `verified_stages`, so the next
    episode replayed the previous one's scored stages, arrived pre-SETTLED, and produced
    the score series `0, 66.7, 0, 0, 0`.
    -> DIAL keeps EVERY piece of per-episode state inside the thread-local `_Ctx`.  Module
       globals hold counters and the report path only.  Nothing replayable lives outside
       the ctx, so the defect has no place to occur.

What DIAL does instead: it injects a compact *diagnosis* block into the PLANNER's message
list (`ApiMemoryPlanner._build_messages`, the same channel GPM's digest and BOLT's board
use — planner-facing and thread-local).  The planner renders its stage primitive as it
always does, and that primitive reaches the VLA unchanged.

Credit arrives at the observer hook.  `override_vla_prompt(..., stage_done={name: bool})`
is the only place in the stack that reports "this stage is now credited" — the reward
channel the architecture needs and that no archived arm reads.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from .policy import A_QUERY, A_RESAMPLE, A_RESYNC, DIALPolicy
from .strategy import StrategyBank
from .types import family_of

_MOD = "harness.api_vlm_planner"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_PRIOR_TASKS = [5, 8, 19, 22]

# Module-level: counters and paths ONLY.  Never per-episode replayable state (defect 3).
_STATE: dict[str, Any] = {
    "installed_at": None,
    "errors": [],
    "n_episodes": 0,
    "n_build": 0,
    "n_stage_obs": 0,
    "n_report": 0,
    "n_credit_events": 0,
    "report_every": 25,
    "stage_names": [],
    "prior_records": 0,
    "prior_source": "",
}
_LOCAL = threading.local()


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in _TRUTHY


def dial_on() -> bool:
    return _truthy(os.environ.get("MEMEXP_DIAL"))


# ---------------------------------------------------------------------------------------
# Thread-local per-episode context
# ---------------------------------------------------------------------------------------
class _Ctx:
    """All per-episode state.  Nothing that can be replayed lives in a module global."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.bank = StrategyBank()
        self.policy = DIALPolicy(bank=self.bank)
        self.seeded = False
        self.step = 0
        self.obligation = ""
        self.family = "other"
        self.obj = "the target object"
        self.credited: set[str] = set()
        self.last_action = ""
        self.zero_displacement = False
        self.retract_or_slip = False
        self.no_observation = False
        self.closed = False


def _live_ctx() -> _Ctx | None:
    ctx = getattr(_LOCAL, "ctx", None)
    if ctx is None or ctx.closed:
        return None
    return ctx


def _new_ctx(planner: Any) -> _Ctx:
    prev = getattr(_LOCAL, "ctx", None)
    if prev is not None:
        prev.closed = True
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


# ---------------------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------------------
def _stage_names_for(planner: Any) -> list[str]:
    try:
        task_info = getattr(planner, "task_info", None)
        task_id = int(getattr(task_info, "task_id", 0) or 0)
        if not task_id:
            return []
        import importlib
        stage_mod = importlib.import_module("task2_26_reference_stage")
        return [str(getattr(s, "name", "") or "") for s in stage_mod._task_specs(task_id)]
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"stage names: {exc!r}")
        return []


def _object_of(stage_name: str) -> str:
    """`01_Lift_Tomato_Sauce` -> `tomato sauce`.  Used only to render a directive."""
    parts = [p for p in str(stage_name or "").split("_") if p]
    if len(parts) >= 3:
        return " ".join(parts[2:]).lower()
    if len(parts) >= 2:
        return parts[-1].lower()
    return "the target object"


def _warm_bank(bank: StrategyBank) -> None:
    """Fold in the archived reward stream once per process.  Failure is non-fatal."""
    if bank.n_records > 0:
        return
    try:
        from pathlib import Path
        from .attribution import extract_all
        root = Path(__file__).resolve().parent.parent / "results"
        recs = extract_all(root, tasks=_PRIOR_TASKS)
        if recs:
            bank.warm(recs)
            _STATE["prior_records"] = bank.n_records
            _STATE["prior_source"] = str(root)
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"warm prior: {exc!r}")


def _seed(ctx: _Ctx) -> None:
    names = _stage_names_for(ctx.planner)
    if names:
        _STATE["stage_names"] = names
    _warm_bank(ctx.bank)
    ctx.seeded = True


def _set_stage(ctx: _Ctx, name: str) -> None:
    if not name or name == ctx.obligation:
        return
    ctx.obligation = name
    ctx.family = family_of(name)
    ctx.obj = _object_of(name)
    ctx.policy.set_obligation(name, ctx.family, ctx.obj)


# ---------------------------------------------------------------------------------------
# Directive rendering (planner channel; never the VLA prompt)
# ---------------------------------------------------------------------------------------
def render_directive(ctx: _Ctx, decision) -> str:
    """Compact diagnosis block for the PLANNER channel.

    Deliberately contains NO rendered attempt sentence.  BOLT v8 established the cost of
    printing something the planner can copy: the board printed `[n3] grasp tomato sauce
    [OPEN]`, the planner echoed it verbatim, the robot grasped without lifting, the scored
    `01_Lift_*` stage never fired, and three seeds scored 0.0.  DIAL prescribes the
    attempt as a CLASS to be rendered, and says so explicitly.
    """
    lines = [
        "[DIAL diagnosis]",
        f"obligation={ctx.obligation or 'unknown'}  family={ctx.family}",
        f"bottleneck={decision.bottleneck} "
        f"P={decision.beta.get(decision.bottleneck, 0.0):.2f} "
        f"lambda_hat={ctx.policy.belief.lambda_hat:.0f} "
        f"attempts={ctx.policy.attempts.get(ctx.family, 0)}",
    ]
    top = sorted(decision.beta.items(), key=lambda kv: -kv[1])[:2]
    lines.append("posterior=" + " ".join(f"{k}:{v:.2f}" for k, v in top))
    if decision.action == A_RESYNC:
        lines.append(
            "directive=HOLD. Credit for the current attempt has not arrived and the attempt "
            "may still be in flight. Keep the current aim and repeat it; do not change the "
            "instruction.")
    elif decision.action == A_QUERY:
        lines.append(
            "directive=OBSERVE. The bottleneck is perceptual or referential, not motor. "
            "Consult memory for the target's last observed position before naming it, and "
            "state the position you are using.")
    elif decision.action == A_RESAMPLE and decision.strategy is not None:
        s = decision.strategy
        lines.append(
            f"directive=NEW ATTEMPT CLASS. The previous attempt class for this obligation "
            f"did not earn credit after the full expected delay. Render this attempt: "
            f"verb={s.verb_form} approach={s.approach} hold={s.hold_steps}.")
    else:
        lines.append(
            "directive=CONTINUE. Re-issue the current attempt for this obligation, in the "
            "stage's own terms. Do not change the aim: the delay before credit arrives is "
            "long (measured median 25 steps, p90 130), and repeating the attempt is what "
            "earns the stage.")
    cur = ctx.policy.last_key.get(ctx.family)
    if cur is not None:
        lines.append("previous_attempt_class=" + "/".join(str(x) for x in cur))
    lines.append(f"[reason] {decision.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------------------
def _wrap_episode_end(module) -> None:
    """The REAL credit channel.

    DEFECT 14: an earlier version read credit from the `stage_done` argument of
    `override_vla_prompt`.  That argument is NOT a credit signal — it is a placeholder
    for the stages *before the current index* and is built everywhere as
    `{spec.name: False for spec in stage_specs[:stage_idx]}`, i.e. always all-False (see
    `harness/controller.py`).  Reading it meant `n_credit_events == 0` for the whole run:
    the scheduler never received a single reward, so it could never learn that an attempt
    was good, and it resampled forever.  A live run's own report proves it:
    `n_stage_obs=375` but `n_credit_events=0`.

    The signal that carries real values is `HarnessController.on_episode_end`, which the
    evaluator calls with the true `stage_done` mapping and `stage_score_pct`.
    """
    cls = getattr(module, "HarnessController", None)
    orig = getattr(cls, "on_episode_end", None) if cls is not None else None
    if orig is None or getattr(orig, "_dial_end_wrapped", False):
        return

    def wrapper(self, *args, **kwargs):
        try:
            ctx = _live_ctx()
            if dial_on() and ctx is not None:
                raw = kwargs.get("stage_done")
                if raw is None and len(args) > 3:
                    raw = args[3]
                done = {str(k) for k, v in (raw or {}).items() if v}
                for stage in sorted(done):
                    ctx.policy.on_credit(stage, step=ctx.step)
                # Families that earned nothing this episode: their outstanding attempt failed.
                credited_fams = {family_of(s) for s in done}
                for fam in list(ctx.policy.pending):
                    if fam not in credited_fams:
                        ctx.policy.retire(family=fam)
                ctx.credited |= done
                _STATE["n_credit_events"] = int(_STATE.get("n_credit_events") or 0) + len(done)
                _write_report(ctx)
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"].append(f"episode_end: {exc!r}")
        return orig(self, *args, **kwargs)

    wrapper._dial_end_wrapped = True  # type: ignore[attr-defined]
    cls.on_episode_end = wrapper


def _wrap_harness_stage(module=None) -> None:
    """Observer only.  MUST return the original prompt (defect 1).

    Takes the module rather than importing it, because `harness` is NOT necessarily
    importable when `install()` runs: `arms/dial.sh` puts only this experiment's `pysite`
    on PYTHONPATH ahead of whatever the submitting shell happened to export, and the
    evaluator adds `evaluation_benchmark` to `sys.path` at run time.  A direct
    `from harness import controller` therefore raises `ModuleNotFoundError`, and an earlier
    version of this file swallowed that and produced a run with NO binding at all: the
    hooks were never installed, the archived evidence was empty, and the arm scored 0,0,0
    looking exactly like an architectural failure.  Always bind from the module object.
    """
    if module is None:
        try:
            from harness import controller as module  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"].append(f"stage hook import: {exc!r}")
            return
    cls = getattr(module, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None) if cls is not None else None
    if orig is None or getattr(orig, "_dial_stage_wrapped", False):
        return

    def wrapper(self, base_prompt, *, stage_idx=0, stage_specs=None, stage_done=None,
                step=0, **extra):
        if dial_on():
            try:
                _STATE["n_stage_obs"] = int(_STATE.get("n_stage_obs") or 0) + 1
                ctx = _live_ctx()
                if ctx is not None:
                    names = [getattr(s, "name", "") for s in (stage_specs or [])]
                    i = int(stage_idx)
                    if 0 <= i < len(names) and names[i]:
                        _set_stage(ctx, str(names[i]))
                    if isinstance(stage_done, dict):
                        done = {str(k) for k, v in stage_done.items() if v and str(k).strip()}
                        fresh = done - ctx.credited
                        if fresh:
                            ctx.credited |= done
                            _STATE["n_credit_events"] = int(
                                _STATE.get("n_credit_events") or 0) + len(fresh)
                            for stage in sorted(fresh, key=lambda s: str(s)):
                                ctx.policy.on_credit(stage, step=int(step))
            except Exception as exc:  # noqa: BLE001
                _STATE["errors"].append(f"stage observe: {exc!r}")
        return orig(self, base_prompt, stage_idx=stage_idx, stage_specs=stage_specs,
                    stage_done=stage_done, step=step, **extra)

    wrapper._dial_stage_wrapped = True  # type: ignore[attr-defined]
    cls.override_vla_prompt = wrapper


def _wrap_build_messages(module) -> None:
    """The planner channel: diagnose, then append one block.  Never raises outward."""
    cls = getattr(module, "ApiMemoryPlanner", None)
    if cls is None:
        _STATE["errors"].append("ApiMemoryPlanner missing")
        return
    orig = getattr(cls, "_build_messages", None)
    if orig is None or getattr(orig, "_dial_wrapped", False):
        return

    def _build_messages(self, memory_main_frames, memory_wrist_frames, context_main_frames,
                        context_wrist_frames, *, extra_memory_text=""):
        msgs = orig(self, memory_main_frames, memory_wrist_frames,
                    context_main_frames, context_wrist_frames,
                    extra_memory_text=extra_memory_text)
        if not dial_on():
            return msgs
        try:
            ctx = _get_ctx(self)
            if not ctx.seeded:
                _seed(ctx)
            ctx.step = int(getattr(self, "step", 0) or 0)
            if not ctx.obligation:
                names = list(_STATE.get("stage_names") or [])
                if names:
                    _set_stage(ctx, names[0])
            # one planner step with no credit yet -> age grows, posterior accumulates
            ctx.policy.advance(
                ctx.step,
                zero_displacement=ctx.zero_displacement,
                retract_or_slip=ctx.retract_or_slip,
                no_observation=ctx.no_observation,
            )
            decision = ctx.policy.decide()
            ctx.last_action = decision.action
            # Any decision that names an attempt IS an enactment -- including a repeat.
            # `note_enacted` distinguishes a continuation (same key: refreshes the patience
            # clock only) from a new attempt class (resets both clocks).  Recording only
            # RESAMPLEs, as an earlier version did, meant a held attempt never refreshed its
            # clock and was then judged stale and cancelled by the scheduler itself.
            if decision.strategy is not None:
                ctx.policy.note_enacted(decision.strategy.key(), decision.strategy.text,
                                        family=ctx.family, step=ctx.step)
            if isinstance(msgs, list):
                msgs.append({"type": "text", "text": render_directive(ctx, decision)})
            _STATE["n_build"] = int(_STATE.get("n_build") or 0) + 1
            every = int(_STATE.get("report_every") or 0)
            if every > 0 and _STATE["n_build"] % every == 0:
                _write_report(ctx)
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"].append(f"build_messages: {exc!r}")
        return msgs

    _build_messages._dial_wrapped = True  # type: ignore[attr-defined]
    cls._build_messages = _build_messages


def _wrap_resets(module) -> None:
    """Clear ALL per-episode state at the boundary (defect 3: leave nothing behind)."""
    cls = getattr(module, "ApiMemoryPlanner", None)
    if cls is None:
        return
    for name in ("reset", "reset_episode", "begin_episode", "set_task_info"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, f"_dial_{name}_wrapped", False):
            continue

        def make(o, n):
            def wrapped(self, *a, **k):
                try:
                    ctx = _live_ctx()
                    if ctx is not None and ctx.seeded:
                        _write_report(ctx)
                except Exception:
                    pass
                _LOCAL.ctx = None
                _LOCAL.planner = None
                return o(self, *a, **k)

            setattr(wrapped, f"_dial_{n}_wrapped", True)
            return wrapped

        setattr(cls, name, make(orig, name))


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------
def _write_report(ctx: _Ctx) -> None:
    path = str(os.environ.get("MEMEXP_DIAL_REPORT", "")).strip()
    if not path:
        return
    try:
        report = {
            "installed_at": _STATE.get("installed_at"),
            "errors": list(_STATE.get("errors") or []),
            "obligation": ctx.obligation,
            "family": ctx.family,
            "credited": sorted(ctx.credited),
            "last_action": ctx.last_action,
            "policy": ctx.policy.snapshot(),
            "belief": ctx.policy.belief.snapshot(),
            "bank": ctx.bank.snapshot(),
            "binding": {
                "n_episodes": _STATE.get("n_episodes"),
                "n_build": _STATE.get("n_build"),
                "n_stage_obs": _STATE.get("n_stage_obs"),
                "n_credit_events": _STATE.get("n_credit_events"),
                "stage_names": list(_STATE.get("stage_names") or []),
                "prior_records": _STATE.get("prior_records"),
                "prior_source": _STATE.get("prior_source"),
            },
            # structural: this module never modifies a prompt or a primitive
            "no_rewrite_path": True,
            "actions_available": ["resample", "query", "resync", "persist"],
        }
        with open(f"{path}.{os.getpid()}", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        _STATE["n_report"] = int(_STATE.get("n_report") or 0) + 1
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"report: {exc!r}")


def _write_receipt() -> None:
    """Proof of installation, written even when no planner step ever happens.

    `_write_report` is only reachable from `_build_messages`, so a failed or partial
    install left NO evidence on disk — which is how the run that scored 0,0,0 was
    initially indistinguishable from a genuine architectural failure.  This receipt is
    written at install time and at every successful bind, unconditionally.
    """
    path = str(os.environ.get("MEMEXP_DIAL_REPORT", "")).strip()
    if not path:
        return
    try:
        with open(f"{path}.install.{os.getpid()}", "w", encoding="utf-8") as fh:
            json.dump({
                "installed_at": _STATE.get("installed_at"),
                "pid": os.getpid(),
                "dial_on": dial_on(),
                "bound_modules": sorted(_BOUND),
                "errors": list(_STATE.get("errors") or []),
                "hooks": {
                    "planner_channel": "harness.api_vlm_planner.ApiMemoryPlanner._build_messages",
                    "credit_observer": "harness.controller.HarnessController.on_episode_end",
                    "stage_observer": "harness.controller.HarnessController.override_vla_prompt",
                    "primitive_channel": "NOT WRAPPED (by design)",
                },
            }, fh, indent=1)
    except Exception:
        pass


# Modules DIAL binds.  Bound lazily, by a meta-path finder, so that it does not matter
# whether `harness` is importable at sitecustomize time.  `memexp_evmem_bind.py` uses the
# same mechanism for the same reason.
_TARGETS = ("harness.api_vlm_planner", "harness.controller")
_BOUND: set[str] = set()

# `harness` is a PACKAGE: binding `harness.controller` requires it to exist as a submodule,
# which happens via the package's own import.  Guarding on the loaded module object (rather
# than on sys.modules membership) keeps this correct under partial initialisation.
_SEEN: set[str] = set()


def _bind_module(fullname: str) -> None:
    """Idempotent, never raises.  Called after each target module finishes executing."""
    module = sys.modules.get(fullname)
    if module is None:
        return
    try:
        if fullname == "harness.api_vlm_planner":
            _wrap_build_messages(module)
            _wrap_resets(module)
        elif fullname == "harness.controller":
            _wrap_harness_stage(module)
            _wrap_episode_end(module)
        _BOUND.add(fullname)
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"bind {fullname}: {exc!r}")
    _write_receipt()


class _PostExecLoader:
    """Wrap a module's loader so DIAL binds immediately after it executes."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        if dial_on():
            _bind_module(module.__name__)

    def __getattr__(self, item):
        return getattr(self._wrapped, item)


class _Finder:
    """Bind our target modules whenever they are imported, however late that is."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        from importlib.machinery import PathFinder  # noqa: PLC0415
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        if getattr(spec.loader, "_dial_wrapped", False):
            return None
        spec.loader = _PostExecLoader(spec.loader)
        return spec


def install() -> None:
    """Idempotent.  Never raises: a bind failure must leave the run working."""
    if _STATE["installed_at"] is not None:
        return
    _STATE["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        _STATE["report_every"] = int(os.environ.get("MEMEXP_DIAL_REPORT_EVERY", "25") or 0)
    except Exception:  # noqa: BLE001
        _STATE["report_every"] = 25

    # Lazy binding first: it is the only mechanism that does not depend on `harness` being
    # importable right now (see `_wrap_harness_stage`).
    try:
        if not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"finder install: {exc!r}")

    # Bind anything already imported (the finder only sees FUTURE imports).
    for fullname in _TARGETS:
        if fullname in sys.modules:
            _bind_module(fullname)

    _write_receipt()


__all__ = ["install", "dial_on", "render_directive", "_STATE", "_Ctx", "_LOCAL",
           "_BOUND", "_TARGETS", "_bind_module", "_Finder"]
