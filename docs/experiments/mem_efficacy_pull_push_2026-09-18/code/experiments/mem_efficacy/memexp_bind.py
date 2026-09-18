"""mem_efficacy / attachment for the PULL arm: a small push plus an agent-callable tool loop.

WHAT IT CHANGES, AND THE TWO HOOKS IT USES
------------------------------------------
Both hooks are module attributes the planner resolves at call time, so this module reaches into
one shared file WITHOUT editing it. When `MEMEXP_PULL_ENABLE` is unset every wrapper falls
straight through to the original, so every other arm keeps its measured behaviour -- the same
contract `harness/seam4_bind.py` established, reused here for the same reason.

  * `ApiMemoryPlanner._build_messages`  (prompt construction, once per plan step)
      - observe the PREVIOUS step's primitive into the substrate
      - append the small push block and the tool spec
  * `harness.api_vlm_planner.infer_primitive_via_api`  (the only API call site)
      - if the model answered with a tool call, run it and call again in the SAME conversation

WHY THE LOOP HANGS OFF THE API CALL AND NOT OFF `infer_sync`
------------------------------------------------------------
`infer_sync` is ~1200 lines that build the memory bank, the stage visuals, the PMH ledger and
the trace record before reaching the API. Reimplementing it to insert a loop would fork all of
that, and every future fix upstream would have to be mirrored -- which is how two components
drift. The API call is the narrowest point through which a plan step MUST pass, so wrapping it
leaves the rest of the pipeline byte-identical and makes this module's blast radius one function.

WHY THIS IS NOT THE PMH TOOL ROUNDS
-----------------------------------
PMH's version is a fixed `for _round in range(PMH_MAX_TOOL_ROUNDS)` loop whose DECISION is a
separate API call with its own prompt (`api_vlm_planner.py:3871`). Three consequences, all
measured in this repository:

  1. The round count is fixed before the model has seen anything, so it is either wasted or
     truncating.
  2. The decision call does no planning, so NOTHING IS AT STAKE when it is made; the model
     answers `none` 85% of the time (`pmh_memory.py:2850-2854`).
  3. The retrieved result never re-enters the context that committed to a primitive.

Here the model nominates a primitive and may call a tool IN THE SAME GENERATION, so the need is
real rather than predicted, the termination condition is the model's own primitive, and the
result returns to the conversation that will act on it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from typing import Any

from memexp_substrate import SERVE_MAX

_MOD = "harness.api_vlm_planner"

# Per-thread, because `_build_messages` and the API call that consumes its context both run
# inside the VLM worker, and the same planner object is driven from one worker at a time
# (`VLM_QUEUE_SIZE=1`). A plain module global would be correct only by that accident.
_LOCAL = threading.local()

_STATE: dict[str, Any] = {
    "installed_at": None,
    "exit_report_registered": False,
    "module_patched": False,
    "build_messages_wrapped": False,
    "api_call_wrapped": False,
    "steps": 0,
    "steps_with_push": 0,
    "tool_calls": 0,
    "steps_with_tool": 0,
    "cap_hits": 0,
    "cap_unresolved": 0,
    "primitive_returned": 0,
    "loop_errors": 0,
    "api_errors": 0,
    # Which registry the ARM actually loaded, and why it failed if it did. Recorded in the
    # report so a reader can tell which tool design produced these numbers -- and so a
    # misconfigured module name is a stated failure rather than a silent degradation to the
    # push-only control (see `load_registry`).
    "tools_module": None,
    "tools_module_error": None,
    # Cumulative sub-layer counters. These are ABSORBED from each context as it is retired, so
    # they are process-wide totals rather than a snapshot of whichever episode happened to be
    # live at the last write. Every counter that a verdict depends on has to live here: a
    # per-episode snapshot makes "the Planner called 5 tools" true and "which tools" wrong.
    "substrate_cumulative": {},
    "tools_cumulative": {},
    "by_tool": {},
    "episodes_retired": 0,
    "errors": [],
}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def pull_enabled() -> bool:
    return _truthy(os.environ.get("MEMEXP_PULL_ENABLE"))


def tools_enabled() -> bool:
    """The small push can be run WITHOUT the tool loop, which is the single-variable control for
    'does agent-chosen dereferencing help' given the same pushed content. Keeping it a runtime
    flag rather than a second arm file means the two conditions cannot drift apart."""
    return _truthy(os.environ.get("MEMEXP_PULL_TOOLS", "1"))


def tools_module_name() -> str:
    """Which registry to load. Defaults to the original, so `pullmem` is bit-identical.

    This selector is how the SEARCH arm is added without forking this 600-line binding: the two
    variants differ in the TOOL SET and the SPEC TEXT and in nothing else, so they share one loop.
    Forking the loop would have duplicated the per-PID reporting, the step-identity scoping and the
    fall-through rules -- three things this project has already paid for once each, and three things
    that would then have had two copies to drift.

    A module name is read rather than a boolean because the next variant will certainly want its own
    registry, and an env var that names the module needs no change here when that happens.
    """
    name = str(os.environ.get("MEMEXP_TOOLS_MODULE", "memexp_tools") or "").strip()
    return name or "memexp_tools"


def force_final_text() -> str:
    """The clamp message, taken from the loaded registry when it supplies one.

    It must come from the registry rather than being a constant here, because it is part of the
    output CONTRACT: a registry's finishing form has to name the same fields the planner's own
    instruction names, or the clamp teaches the model a key the parser discards. Hardcoding one
    here would force the same mistake back.
    """
    try:
        text = str(getattr(load_registry(), "FORCE_FINAL", "") or "").strip()
        if text:
            return text
    except Exception:  # noqa: BLE001 - the clamp must never be the thing that breaks a step
        pass
    return _FORCE_FINAL


# What a taught finishing form must parse to, once its placeholders are replaced.
FINISH_SENTINEL = "SENTINEL-PRIMITIVE"
_PLACEHOLDER_STR = re.compile(r'"\s*<[^>]*>\s*"')
_PLACEHOLDER_LIST = re.compile(r"\[\s*<[^>]*>\s*\]")


def concretise_form(text: str) -> str:
    """Replace a taught form's placeholders with concrete values, so the REAL parser can read it.

    This step is not cosmetic, and skipping it made the first version of these assertions VACUOUS.
    A form containing `<your chosen primitive>` is not valid JSON, so `parse_vlm_output` falls
    through to its prose branch and returns the RAW TEXT as the primitive -- which is truthy no
    matter which key name the form used. Measured on the real parser:

        '{"current_primitive": "<...>", "keyframe_positions": [<...>]}'  -> the raw string
        '{"primitive":         "<...>", "keyframe_positions": [<...>]}'  -> the raw string
        '{"current_primitive": "open middle drawer", "keyframe_positions": []}' -> 'open middle drawer'
        '{"primitive":         "open middle drawer", "keyframe_positions": []}' -> ''

    So `bool(parsed)` on the un-substituted form passes for the RIGHT key and for the WRONG one, and
    the assertion cannot detect the defect it exists to detect. After substitution the two cases are
    `FINISH_SENTINEL` and `''`, and the check becomes exact: the parser must read the value of the
    key the PLANNER requires, not merely return something non-empty.
    """
    out = _PLACEHOLDER_LIST.sub("[]", str(text or ""))
    return _PLACEHOLDER_STR.sub(f'"{FINISH_SENTINEL}"', out)


class ToolsModuleError(RuntimeError):    """The configured tool registry could not be loaded or is not usable.

    Raised rather than absorbed because the alternative is invisible: `_build_messages` catches
    everything, so a bad module name would leave the arm running as the PUSH-ONLY control while
    its flags all read as enabled. Downstream that is indistinguishable from "the model declined
    to call the tools" -- which is a finding about the model, and would have been reported as one.
    """


# One import per module name per process. A failure is cached too, so a broken name does not
# re-attempt (and re-print) on every planning step.
_REGISTRY_LOCK = threading.Lock()
_REGISTRY_CACHE: dict[str, Any] = {}


def load_registry():
    """Import and VALIDATE the tool registry named by `MEMEXP_TOOLS_MODULE`.

    Validated here, once, rather than trusted and discovered later: the two things the loop needs
    from a registry are a `MemoryTools` class and the shared `parse_tool_call`, and a module that
    renamed either would otherwise fail per call inside the loop's `except` -- again silently.
    """
    name = tools_module_name()
    with _REGISTRY_LOCK:
        cached = _REGISTRY_CACHE.get(name)
        if cached is not None:
            if isinstance(cached, BaseException):
                raise ToolsModuleError(f"tool registry {name!r} is unusable") from cached
            return cached
        try:
            from importlib import import_module

            mod = import_module(name)
            missing = [a for a in ("MemoryTools", "parse_tool_call") if not hasattr(mod, a)]
            if missing:
                raise AttributeError(f"{name!r} does not export {missing}")
        except Exception as exc:  # noqa: BLE001
            _REGISTRY_CACHE[name] = exc
            _STATE["tools_module_error"] = f"{name}: {exc!r}"
            # An environment marker as well as a counter, because the census runs in a DIFFERENT
            # process from the ones that actually evaluate. The report is the durable channel; the
            # marker is what makes the failure visible to any shell-level audit in between.
            os.environ["MEMEXP_PULL_BROKEN"] = "1"
            sys.stderr.write(
                f"[memexp] FATAL: tool registry {name!r} could not be loaded ({exc!r}); the pull "
                f"arm cannot run its tool loop and would silently degrade to the push-only "
                f"control.\n"
            )
            raise ToolsModuleError(f"tool registry {name!r} is unusable") from exc
        _REGISTRY_CACHE[name] = mod
        _STATE["tools_module"] = name
        return mod


def registry_is_broken() -> bool:
    """Whether a load attempt has already failed. Used to FAIL a run rather than warn about it."""
    return bool(_STATE.get("tools_module_error"))


def _max_rounds() -> int:
    return max(1, int(os.environ.get("MEMEXP_PULL_MAX_ROUNDS", "3")))


def stated_budget(spec_lower: str) -> int:
    """The tool-call budget the SPEC PROMISES, read back out of the spec text.

    Parsed out of the sentence rather than read from the environment, because the check it feeds is
    a comparison between two independent artifacts -- the words the model is shown and the counter
    the loop enforces. A helper that read `MEMEXP_PULL_MAX_ROUNDS` twice could never detect the
    disagreement, which is exactly the defect it exists for: job 595132 ran spec="as many times as
    you need" against fuse=3 and reported `cap_hits=2`.
    """
    m = re.search(r"up to (\d+) tool", str(spec_lower or ""))
    return int(m.group(1)) if m else -1



def _write_report() -> None:
    """Persist this process's counters to a PER-PROCESS file.

    Two defects are fixed here, and the second one is why the first run's numbers were
    unreadable.

    1. A report must be written while the run is happening. Previously the only caller was
       `_bind()`, i.e. install time, so every runtime counter in the file (`steps`,
       `tool_calls`, `n_query`, `n_frames_served_off_context`, `cap_hits`, ...) was frozen at
       zero and read as "the Planner never called a tool" when it meant "nobody ever updated
       this file". Confirmed on job 591479: the surviving file carried
       `"substrate": null, "tools": null`, which is only ever true before a context exists.
       It is now called on every planning step and once at interpreter exit.

    2. The path must be per-process. An arm runs several interpreters against one output
       directory -- `task1` and `tasks2to26` are separate processes, and
       `merge_eval_outputs.sh` runs a `python3` after the evaluation, which loads this module
       via `PYTHONPATH` and would otherwise truncate the real numbers to zeros.
    """
    path = os.environ.get("MEMEXP_PULL_REPORT", "").strip()
    if not path:
        return
    try:
        # Absorb before writing, so the totals below cover every episode this process has run --
        # not only the one that happens to be live at this instant. Done here rather than only at
        # episode boundaries because the last episode never gets a successor to retire it, and
        # the exit-time write is exactly the one that has to be complete.
        _absorb()
        ctx = getattr(_LOCAL, "ctx", None)
        payload = dict(_STATE)
        payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload["pid"] = os.getpid()
        payload["enabled_now"] = pull_enabled()
        payload["tools_enabled_now"] = tools_enabled()
        payload["broken_now"] = registry_is_broken()
        # The live view, for diagnostics: what the CURRENT episode's objects look like.
        payload["substrate"] = ctx.substrate.stats() if ctx else None
        payload["tools"] = ctx.tools.stats() if ctx else None
        # The totals a verdict may be based on. Separate keys, deliberately: the two have
        # different meanings and an archived report cannot be re-interpreted, so the census has to
        # be able to tell which semantics it is reading. `substrate`/`tools` describe ONE episode;
        # these describe the whole process.
        payload["substrate_cumulative"] = dict(_STATE.get("substrate_cumulative") or {})
        payload["tools_cumulative"] = dict(_STATE.get("tools_cumulative") or {})
        payload["by_tool"] = dict(_STATE.get("by_tool") or {})
        payload["substrate_gauges"] = dict(_STATE.get("substrate_gauges") or {})
        own = f"{path}.{os.getpid()}"
        tmp = f"{own}.tmp"

        # Carry forward the transient `substrate`/`tools` snapshots.
        #
        # `_STATE` counters are cumulative, so overwriting this file loses nothing. The substrate
        # and tool statistics are NOT: they are read live from `ctx`, which is null in every write
        # that does not happen mid-plan-step -- including the last one before the process exits.
        # Measured on job 592860: 20 reports, `tool_calls` correct, and `substrate`/`tools` null in
        # ALL of them, so `by_tool`, `n_query`, `n_frames_served_off_context` and "addresses
        # minted" were still unreadable and the census reported "called 5 tools and served 0
        # frames" -- a FAIL drawn from missing data, one layer below the bug this file already
        # fixed. Keeping the last non-null snapshot is what makes the pull's effect readable.
        try:
            with open(own, encoding="utf-8") as handle:
                previous = json.load(handle)
        except Exception:
            previous = {}
        for key in ("substrate", "tools"):
            if payload.get(key) is None and isinstance(previous.get(key), dict):
                payload[key] = previous[key]

        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, own)
    except Exception as exc:  # noqa: BLE001 - a report must never be the reason a run dies
        sys.stderr.write(f"[memexp] could not write pull report {path!r}: {exc!r}\n")


def _register_exit_report() -> None:
    """Write once at exit, so a process that ends without another planning step is still counted.

    Guarded against double registration: `install()` may run in several module identities
    (`sitecustomize` imports it, and the arms' own probes import it again under a different
    name), and an unguarded `atexit` registration would accumulate handlers.
    """
    if _STATE.get("exit_report_registered"):
        return
    import atexit

    atexit.register(_write_report)
    _STATE["exit_report_registered"] = True


class _Ctx:
    """Everything one plan step needs to run its tool loop.

    `snapshot` is the last set of counters absorbed from this context into `_STATE`. It is what
    makes absorption idempotent: the report is written many times per episode and several times
    after the last step, and each absorption takes only the DELTA, so no counter is added twice.
    """

    __slots__ = ("planner", "substrate", "tools", "on_context", "step", "msgs", "snapshot",
                 "closed", "seq")

    def __init__(self, planner, substrate, tools, on_context, step, msgs, seq) -> None:
        self.planner = planner
        self.substrate = substrate
        self.tools = tools
        self.on_context = on_context
        self.step = step
        # The exact list object handed to the API call for this prompt. See `_wrap_api_call`:
        # identity on this object is what scopes the loop to the MAIN planning call.
        self.msgs = msgs
        self.snapshot: dict[str, Any] = {}
        # Set when a reset hook retires this context. `_should_engage` refuses a closed context,
        # which is what stops a retired bank from being queried by a later step.
        self.closed = False
        self.seq = seq


# Cumulative keys, read from the sub-layers. Counters are SUMMED across episodes; gauges are not.
_CUM_SUBSTRATE = (
    "n_action_write", "n_unbound_minted", "n_query", "n_query_hit", "n_query_miss",
    "n_frames_served_off_context", "n_serve_empty_not_yet_observed",
    # Write-path-2 counters. THEY WERE MISSING FROM THIS TUPLE, and that is the whole reason
    # `substrate_cumulative` reported `n_spec_minted = 0` on a run whose live substrate reported
    # 12 (job 593817, task 4: facts=13, n_spec_unopened=12). The absorption loop iterates THIS
    # tuple, so a counter absent from it is never folded in and reads as a confident zero --
    # while the census then FAILs the arm for a defect that does not exist. Adding a counter to
    # `Substrate.stats()` is not enough; this tuple is the second place it must appear, and the
    # two drifting apart produces exactly the false verdict this project keeps having to retract.
    "n_spec_minted", "n_stage_active", "n_stage_verified",
)
_CUM_TOOLS = (
    "n_calls", "n_repeat_calls", "n_frames_served", "n_spec_shown",
    "n_search", "n_search_hit", "n_search_miss",
)
# A gauge describes a moment. Summing `unbound_now` across 42 episodes would report an address
# count no episode ever had, and `max` is taken instead so the number still means something.
# `n_spec_unopened` is a gauge for the same reason: it counts the seeded addresses whose window is
# still shut RIGHT NOW, so summing across episodes would report a backlog no episode ever had.
_GAUGE_SUBSTRATE = ("facts", "addresses", "unbound_now", "n_spec_unopened")

# Every context this process has created, whether or not the thread that made it is still the
# thread reading this. Module-level on purpose -- see `_reset_ctx`.
_CTX_REGISTRY: list[_Ctx] = []
_CTX_LOCK = threading.Lock()
_CTX_SEQ = 0


def _bump(bucket: str, key: str, n: int) -> None:
    if n:
        d = _STATE.setdefault(bucket, {})
        d[key] = int(d.get(key) or 0) + int(n)


def _absorb_one(ctx: _Ctx) -> None:
    """Fold one context's sub-layer counters into `_STATE`, taking only the delta.

    The snapshot is nested (`{"substrate": ..., "tools": ...}`) rather than flat with a key prefix.
    The first version flattened it and then read the delta from the UN-prefixed key, so the previous
    value was never found and every absorption re-added the full count -- caught by this module's
    own idempotence assertion, which exists because the report is written after every planning step
    and again at exit, so a non-idempotent absorption makes the totals depend on how many times the
    file happened to be written.
    """
    prev = ctx.snapshot or {}
    prev_sub = prev.get("substrate") or {}
    prev_tools = prev.get("tools") or {}
    try:
        sub_now = dict(ctx.substrate.stats())
    except Exception:  # noqa: BLE001
        sub_now = {}
    try:
        tools_now = dict(ctx.tools.stats())
    except Exception:  # noqa: BLE001
        tools_now = {}

    for k in _CUM_SUBSTRATE:
        _bump("substrate_cumulative", k, int(sub_now.get(k) or 0) - int(prev_sub.get(k) or 0))
    for k in _CUM_TOOLS:
        _bump("tools_cumulative", k, int(tools_now.get(k) or 0) - int(prev_tools.get(k) or 0))
    # `by_tool` is the number the census needs to tell the SEARCH tool from the dereference tool,
    # and it is exactly the number that was wrong while it was read live from a per-episode
    # object: the last episode of a 26-task run usually has no tool calls at all, so a run with
    # twenty calls reported `by_tool = {}`.
    prev_by = prev_tools.get("by_tool") or {}
    for name, n in (tools_now.get("by_tool") or {}).items():
        _bump("by_tool", str(name), int(n) - int(prev_by.get(name) or 0))

    ctx.snapshot = {"substrate": sub_now, "tools": tools_now}

    for k in _GAUGE_SUBSTRATE:
        try:
            gauges = _STATE.setdefault("substrate_gauges", {})
            gauges[k] = max(int(gauges.get(k) or 0), int(sub_now.get(k) or 0))
        except (TypeError, ValueError):
            pass


def _absorb() -> None:
    """Absorb every registered context. Idempotent, so it is safe to call on every write."""
    with _CTX_LOCK:
        ctxs = list(_CTX_REGISTRY)
    for ctx in ctxs:
        try:
            _absorb_one(ctx)
        except Exception as exc:  # noqa: BLE001 - a report may never be the reason a run dies
            _STATE.setdefault("errors", []).append(f"absorb: {exc!r}")


def _live_ctx() -> _Ctx | None:
    ctx = getattr(_LOCAL, "ctx", None)
    return ctx if ctx is not None and not ctx.closed else None


# ---------------------------------------------------------------------------------------
# Hook 4: the ACTIVE STAGE, taken from the evaluator's ground-truth stage machine.
# ---------------------------------------------------------------------------------------
def _wrap_harness_stage() -> None:
    """Record the active stage from `HarnessController.override_vla_prompt`.

    THIS REPLACES A BROKEN DRIVER, and the failure it replaces is worth stating precisely because
    the arm still reported itself healthy throughout. The task-spec seed opens each address's
    evidence window when its STAGE activates, and the first version of that driver polled
    `planner.episodic_store.current_stage_name`. Job 593817 showed the consequence on task 4:

        n_spec_minted = 12     n_spec_unopened = 12     n_stage_active = 0

    The seed ran and the addresses existed, but no window ever opened, so all twelve were
    advertised in the push block every step AND permanently unreadable. `episodic_store` was
    None the whole time: it comes from `create_store_if_needed`, which returns None unless the
    proactive-memory config is enabled, and this arm runs `PROACTIVE_MODE=off`. The driver was
    reading a field of an object the arm never constructs.

    The evaluator does hold the ground truth, but not there. It keeps `stage_idx` and `stage_specs`
    in its own loop and hands both to the harness on EVERY step:

        prompt_for_vla = harness.override_vla_prompt(..., stage_idx=stage_idx,
                                                     stage_specs=stage_specs, ...)

    (eval_fullvlm26_async_vlm_vla.py:1700). That call site is also the one the stage SCORER uses,
    so the stage named here is the same one whose predicate gates the score -- not a second
    opinion about where the episode is. Recording it here is therefore strictly better than
    polling any store: it cannot be None, it cannot drift from the scorer, and it is updated
    regardless of which optional subsystems the arm has turned off.

    A hook rather than a new planner attribute because the harness is constructed by the
    evaluator, not the planner, so the planner has no reference to reach it from.
    """
    try:
        from harness import controller as _ctl
    except Exception as exc:  # noqa: BLE001 - the arm must still run without this
        _STATE["stage_hook_error"] = f"import harness.controller: {exc!r}"
        return
    cls = getattr(_ctl, "HarnessController", None)
    orig = getattr(cls, "override_vla_prompt", None) if cls is not None else None
    if orig is None:
        _STATE["stage_hook_error"] = "HarnessController.override_vla_prompt not found"
        return
    if getattr(orig, "_memexp_stage_wrapped", False):
        return

    def wrapper(self, base_prompt, *, stage_idx=0, stage_specs=None, stage_done=None,
                step=0, **extra):
        try:
            names = [getattr(s, "name", "") for s in (stage_specs or [])]
            i = int(stage_idx)
            if 0 <= i < len(names) and names[i]:
                _STATE["active_stage"] = str(names[i])
                _STATE["active_stage_idx"] = i
                _STATE["active_stage_step"] = int(step)
                _STATE["stage_hook_calls"] = int(_STATE.get("stage_hook_calls", 0)) + 1
                # The durable evidence that this driver is live. `n_stage_active` only counts
                # OPENINGS, so it stays 0 for a task whose first stage is opened by the seed-time
                # default -- this counter distinguishes "the hook never fired" from "the hook
                # fired and no stage changed".
                _STATE["stage_names_seen"] = sorted(
                    set(_STATE.get("stage_names_seen") or []) | {str(names[i])}
                )
            # PMH verified-completion write path. `stage_done` is keyed by stage NAME (see
            # `eval_common.run_episode_with_stages`) and is the same dict the stage SCORER
            # consults; a True entry here is therefore a ground-truth fact, not a Planner guess.
            # Applied to the substrate in `_build_messages` (the harness has no substrate of its
            # own).
            if isinstance(stage_done, dict):
                done = {str(k) for k, v in stage_done.items() if v and str(k).strip()}
                if done:
                    _STATE["verified_stages"] = sorted(
                        set(_STATE.get("verified_stages") or []) | done
                    )
        except Exception:  # noqa: BLE001 - an observability hook must never break the step
            pass
        return orig(self, base_prompt, stage_idx=stage_idx, stage_specs=stage_specs,
                    stage_done=stage_done, step=step, **extra)

    wrapper._memexp_stage_wrapped = True
    cls.override_vla_prompt = wrapper
    _STATE["stage_hook_installed"] = True


def _current_stage(planner) -> str:
    """The stage the episode is actually in, from the best available source.

    Order is deliberate: the evaluator's stage machine first, because it is ground truth and is
    always present when the harness is enabled; the episodic store second, because PMH-style arms
    do build one and it is correct there. Neither being available leaves the stage empty, which
    closes no windows and is reported by the census as `n_stage_active == 0` rather than passing
    silently.
    """
    s = str(_STATE.get("active_stage") or "").strip()
    if s:
        return s
    return str(getattr(getattr(planner, "episodic_store", None), "current_stage_name", "") or "").strip()



def _seed_from_task_spec(planner, substrate) -> int:
    """Expand the task's stage list into substrate addresses. Never raises.

    The stage list comes from `task2_26_reference_stage._task_specs(task_id)`, the SAME function
    the evaluator uses to score stages, so the two cannot disagree about what the task contains.
    Re-deriving it from the task description text instead would create a second source of truth
    that would drift -- and the drift would be invisible, because a substrate that seeds the wrong
    addresses still reports a healthy write path.

    Imported lazily and wrapped: this runs inside a plan step, and a substrate must never be able
    to fail an episode. A failure is recorded in `_STATE` and reported by the census as a FAIL,
    which distinguishes "the seed did not run" from "the seed ran and the Planner ignored it".
    """
    try:
        # Cleared FIRST, so this field always describes the MOST RECENT attempt. It is process-wide
        # (`_STATE`) while the substrate is per-episode, so without this the flag is sticky: any
        # earlier episode that could not seed -- a probe planner with no `task_info`, or a genuine
        # failure on task 3 -- would keep reporting a failure on every later episode that seeded
        # perfectly. `selftest_pull.py` G0b-10 caught exactly that, on the one gate whose whole
        # assertion is that this field is empty.
        _STATE["seed_error"] = None
        task_info = getattr(planner, "task_info", None)
        task_id = int(getattr(task_info, "task_id", 0) or 0)
        if not task_id:
            _STATE["seed_error"] = "no task_id on planner.task_info"
            return 0
        import importlib

        stage_mod = importlib.import_module("task2_26_reference_stage")
        names = [getattr(s, "name", "") for s in stage_mod._task_specs(task_id)]
        n = substrate.seed_from_stages(names, 0)
        _STATE["seeded_task_id"] = task_id
        _STATE["seeded_stages"] = len(names)
        _STATE["seeded_addresses"] = int(_STATE.get("seeded_addresses", 0)) + int(n)
        return int(n)
    except Exception as exc:  # noqa: BLE001 - a substrate must not be able to crash a plan_step
        _STATE["seed_error"] = repr(exc)
        return 0


def _get_ctx(planner):
    """The ONE context for this episode, created when the episode changes.

    One per episode, not one per plan step. The context is the tracking unit for the delta
    accounting, so two contexts wrapping the same substrate would each absorb the same counters
    and every total would be doubled. Its `step`, `on_context` and `msgs` fields are MUTATED by
    `_build_messages` on each step instead.

    FRESHNESS IS A DISJUNCTION, and the second half is the load-bearing one. Keying only on planner
    IDENTITY is wrong because the evaluator builds ONE planner and reuses it for all 26 tasks: the
    identity never changes, so the bank would never be replaced and task 26 would be answered partly
    out of tasks 1-25. That is not a leak of style -- it is a confound that would make every
    per-task score meaningless while every flag still read as enabled. The measurement suggesting
    identity was sufficient (42 contexts created in one run) came from the VLM worker being a FRESH
    THREAD per episode, which resets a thread-local for reasons that have nothing to do with the
    episode boundary. Relying on that is relying on an implementation detail of the evaluator's
    thread pool. So a retired context also forces a rebuild, which is what makes the reset hooks
    load-bearing instead of merely non-destructive.
    """
    if getattr(_LOCAL, "planner", None) is not planner or _live_ctx() is None:
        global _CTX_SEQ

        from memexp_substrate import Substrate

        # Retire the outgoing context BEFORE overwriting the reference, so its counters reach the
        # cumulative totals. This is the only guaranteed place to do it: the thread that replaces
        # a context is the thread that owns it, and it runs even when the reset hooks do not
        # (which is the measured case -- `resets` was absent from every archived report).
        previous = getattr(_LOCAL, "ctx", None)
        if previous is not None:
            previous.closed = True
            _absorb_one(previous)

        sub = Substrate()
        _LOCAL.planner = planner
        _LOCAL.substrate = sub
        # Write path 2, seed. Done HERE rather than in the `set_task_info` hook because the
        # substrate is created LAZILY at the first plan step of the episode: at hook time the new
        # bank does not exist yet, and the hook that retires the old one cannot seed the new one.
        # Seeding at the only place the bank is constructed makes it impossible for an episode to
        # begin with an unseeded bank -- which is the shape of defect ("enabled but empty") this
        # project has now paid for four times.
        #
        # The stage list is a pure function of the task id, so this is a model-free, deterministic
        # expansion of the task skeleton: the Planner no longer has to discover which containers
        # the task involves. See `Substrate.seed_from_stages` for the measured defect it closes.
        _seed_from_task_spec(planner, sub)
        # The stage recorded by the harness hook describes the PREVIOUS episode until the eval
        # loop's next step, so it must be cleared or the new episode would immediately open the
        # addresses of whatever stage the last task ended on. Cleared here rather than in the
        # reset hooks because the hooks can be missed (see `_get_ctx`) while this line runs
        # whenever a bank is built.
        _STATE["active_stage"] = ""
        _STATE["active_stage_idx"] = -1
        _STATE["verified_stages"] = []
        # The registry is resolved and VALIDATED here, at the first step of the episode, so a bad
        # `MEMEXP_TOOLS_MODULE` fails loudly at the start rather than per call inside the loop.
        registry = load_registry()
        tools = registry.MemoryTools(sub)
        _LOCAL.tools = tools
        _LOCAL.tools_module = tools_module_name()
        _CTX_SEQ += 1
        ctx = _Ctx(planner, sub, tools, set(), 0, None, _CTX_SEQ)
        _LOCAL.ctx = ctx
        with _CTX_LOCK:
            _CTX_REGISTRY.append(ctx)
        _STATE["episodes_started"] = int(_STATE.get("episodes_started", 0)) + 1
    return _LOCAL.substrate, _LOCAL.tools


def _reset_ctx(why: str) -> None:
    """Retire every live context on an episode boundary.

    THREAD-AGNOSTIC, which the previous version was not. The reset hooks run on the evaluator's
    main thread while the planning context is created inside the VLM worker, and `_LOCAL` is
    thread-local -- so `getattr(_LOCAL, "planner", None)` was None there and the function's whole
    body was skipped. Measured: `resets` is absent from every archived `memexp_pull_report.json`,
    including the one for job 592860, while `reset_hooks` shows both hooks were installed.

    It also no longer CLEARS anything. It used to call `tools.calls.clear()`, which destroys the
    very counters the census reads; retiring a named context and absorbing its totals is both
    sufficient and non-destructive.
    """
    retired = 0
    with _CTX_LOCK:
        for ctx in _CTX_REGISTRY:
            if not ctx.closed:
                ctx.closed = True
                try:
                    _absorb_one(ctx)
                except Exception as exc:  # noqa: BLE001
                    _STATE.setdefault("errors", []).append(f"reset absorb: {exc!r}")
                retired += 1
    _STATE["episodes_retired"] = int(_STATE.get("episodes_retired", 0)) + retired
    _STATE.setdefault("resets", []).append({"why": why, "retired": retired})


# ---------------------------------------------------------------------------------------
# Hook 1: prompt construction -- observe the previous action, push a small block
# ---------------------------------------------------------------------------------------
def _wrap_build_messages(module) -> None:
    cls = module.ApiMemoryPlanner
    orig = cls._build_messages
    if getattr(orig, "_memexp_wrapped", False):
        return

    def _build_messages(self, memory_main_frames, memory_wrist_frames, context_main_frames,
                        context_wrist_frames, *, extra_memory_text=""):
        # PrediMem bank is built by the planner when `VLM_USE_KEYFRAME_MEMORY=1`, but THIS arm's
        # contract is that historical frames arrive only on demand. Strip them BEFORE the
        # official builder emits them, so a single call produces the pull-shaped prompt and
        # `n_frames_served` stays a gain metric rather than a volume metric.
        if pull_enabled():
            memory_main_frames, memory_wrist_frames = [], []
        msgs = orig(
            self,
            memory_main_frames,
            memory_wrist_frames,
            context_main_frames,
            context_wrist_frames,
            extra_memory_text=extra_memory_text,
        )
        if not pull_enabled():
            return msgs
        try:
            substrate, tools = _get_ctx(self)

            # The recent window is the frames currently on context. `infer_sync` sets
            # `self.step = step_idx + 1`, so step_idx == self.step - 1 and the window starts at
            # `self.step - len(context_main_frames)`. Subtracting it from what a pull may serve
            # is what keeps `n_frames_served` interpretable: a frame the Planner can already see
            # is not evidence, and re-serving it is the redundancy that made PMH's `picked=1`
            # against `n_gain_skipped=101`.
            #
            # K_indices are NOT added to on_context: they were stripped above, so they must stay
            # available for tool serve. Marking them on-context while absent from the prompt was
            # the previous defect that made the PrediMem bank look "used" while every pull of it
            # returned nothing new.
            n_ctx = len(context_main_frames)
            recent_start = max(0, int(self.step) - n_ctx)
            on_context = set(range(recent_start, recent_start + n_ctx))

            # Write path. `self._current_subtask` is the PREVIOUS step's output at this moment,
            # because the prompt is built before this step's primitive exists. That is exactly
            # the right thing to record: it is the most recent thing the robot has done.
            #
            # The frame store is handed to the substrate here rather than being passed through
            # the tool call, because frames are resolved at READ time (see the substrate header)
            # and the substrate needs to know which frames exist by then.
            substrate.frame_store = self.frame_store_main
            # PrediMem retrieval bank: prefer nominated/anchor frames when a tool pulls.
            substrate.set_retrieval_bank(list(getattr(self, "K_indices_abs", []) or []))
            prev = str(getattr(self, "_current_subtask", "") or "").strip()
            step_of_prev = max(0, int(self.step) - 1)
            if prev:
                substrate.note_action(prev, step_of_prev)

            # Write path 2 -- the harness's stage machine. See `_wrap_harness_stage` for why the
            # source is the evaluator's own `stage_idx` and NOT `episodic_store`, which is None in
            # this arm (job 593817: n_spec_minted=12, n_stage_active=0, PROACTIVE_MODE=off).
            #
            # `_current_stage` reads the stage named by the same call that gates the stage SCORER,
            # so an opened window is anchored to a verified transition rather than to the
            # Planner's opinion of where it is.
            stage_now = _current_stage(self)
            if stage_now and stage_now != getattr(substrate, "_active_stage", ""):
                substrate.note_stage_active(stage_now, step_of_prev)
                _STATE["stage_active_events"] = int(_STATE.get("stage_active_events", 0)) + 1

            # Write path 3 -- verified stage completions (PMH ground-truth facts).
            for nm in list(_STATE.get("verified_stages") or []):
                substrate.note_verified_stage(nm, step_of_prev)

            push = substrate.render_push(on_context=on_context)
            _STATE["steps"] = int(_STATE["steps"]) + 1
            if push:
                _STATE["steps_with_push"] = int(_STATE["steps_with_push"]) + 1
                msgs.append({"type": "text", "text": push})
            if tools_enabled():
                msgs.append({"type": "text", "text": tools.spec_text()})

            tools.reset_step_cache()
            # Mutate the episode's single context instead of allocating a new one per step. A new
            # one would start with an empty snapshot, its first absorption would take the FULL
            # running totals, and the previous context would have already absorbed them -- every
            # counter doubled from the second step of the episode onward.
            ctx = getattr(_LOCAL, "ctx", None)
            if ctx is not None:
                ctx.on_context = on_context
                ctx.step = int(self.step)
                ctx.msgs = msgs
                ctx.tools = tools
            # The step counter is the one that proves the hook is engaged at all. Flushing it
            # here means "0 steps" in the artifact is a real finding ("this hook never ran"),
            # which it could not be while the file was written once at install time.
            _write_report()
        except Exception as exc:  # noqa: BLE001 - a memory layer may not break the planner
            # A registry that could not be loaded is separated from every other failure on
            # purpose. It means the arm is running as the PUSH-ONLY control, so its score is not a
            # measurement of the pull at all -- and the census needs to say FAIL rather than the
            # WARN ("the Planner never called a tool") that this would otherwise produce.
            if isinstance(exc, ToolsModuleError):
                _STATE["tools_module_error"] = str(_STATE.get("tools_module_error") or exc)
            _STATE["loop_errors"] = int(_STATE["loop_errors"]) + 1
            _STATE["errors"].append(f"build_messages: {exc!r}")
            _LOCAL.ctx = None
        return msgs

    _build_messages._memexp_wrapped = True
    cls._build_messages = _build_messages
    _STATE["build_messages_wrapped"] = True


def _should_engage(user_content):
    """The single place that decides whether a given API call runs the tool loop.

    Factored out so the scoping rule can be tested directly rather than only inferred from a
    26-task run. `infer_primitive_via_api` has four call sites in the planner and only the main
    planning call consumes the list `_build_messages` just produced, so identity on that list is
    what scopes the loop. Matching on identity means a stale context can never wrap the
    stage-boundary call or a PMH/SEAM tool-decision call -- a `ctx is not None` test would.
    """
    if not pull_enabled() or not tools_enabled():
        return None
    # `_live_ctx()` rather than a raw read: a context retired by a reset hook must not serve a
    # later step's retrieval, or a new episode would be answered out of the previous bank.
    ctx = _live_ctx()
    if ctx is None or ctx.msgs is not user_content:
        return None
    return ctx


# ---------------------------------------------------------------------------------------
# Hook 2: the API call -- run the tool loop in one conversation
# ---------------------------------------------------------------------------------------
_FORCE_FINAL = (
    "TOOLS ARE NOW CLOSED for this planning step. Do not output a tool call. "
    "Output exactly one JSON object with exactly two fields: "
    '{"current_primitive": "<your chosen primitive>", "keyframe_positions": [<positions>]}.'
)


def _wrap_api_call(module) -> None:
    orig = module.infer_primitive_via_api
    if getattr(orig, "_memexp_wrapped", False):
        return

    def infer_primitive_via_api(*, system_prompt, user_content, **kw):
        ctx = _should_engage(user_content)
        if ctx is None:
            return orig(system_prompt=system_prompt, user_content=user_content, **kw)

        # From the SAME validated registry the arm loaded, not from a hardcoded module name.
        # Importing `memexp_tools` here would make the "registry-independent loop" claim false and
        # would break the moment a registry moved its parser.
        parse_tool_call = load_registry().parse_tool_call

        planner = ctx.planner
        known = set(ctx.tools.tools)
        # Copy: this list is the caller's, and `_build_messages` handed it over for one call.
        # Mutating it would leak this step's tool results into any later use of the same list.
        msgs = list(user_content)
        base_kw = dict(kw)
        step_used_tool = False

        try:
            for round_i in range(_max_rounds() + 1):
                out = orig(system_prompt=system_prompt, user_content=msgs, **base_kw)
                if out is None:
                    _STATE["api_errors"] = int(_STATE["api_errors"]) + 1
                    return None
                call = parse_tool_call(out, known)
                if call is None:
                    _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                    return out

                if round_i >= _max_rounds():
                    # The fuse. Reached only if the model keeps calling tools past the cap, which
                    # means either the cap is too small or a tool result is not answering. Both
                    # are counted rather than absorbed, because the alternative -- returning the
                    # tool-call text as if it were a primitive -- silently yields no primitive and
                    # costs real score for a reason that looks like "the Planner produced nothing".
                    _STATE["cap_hits"] = int(_STATE["cap_hits"]) + 1
                    msgs.append({"type": "text", "text": f"Your output was:\n{out}"})
                    msgs.append({"type": "text", "text": force_final_text()})
                    forced = orig(system_prompt=system_prompt, user_content=msgs, **base_kw)
                    if forced is not None and parse_tool_call(forced, known) is None:
                        _STATE["primitive_returned"] = int(_STATE["primitive_returned"]) + 1
                        return forced
                    _STATE["cap_unresolved"] = int(_STATE["cap_unresolved"]) + 1
                    _STATE["errors"].append(
                        f"cap unresolved at step {ctx.step}: model would not stop calling tools"
                    )
                    return forced if forced is not None else out

                args = dict(call.get("args") or {})
                args["_on_context"] = ctx.on_context
                result = ctx.tools.call(call["tool"], args)
                _STATE["tool_calls"] = int(_STATE["tool_calls"]) + 1
                if not step_used_tool:
                    _STATE["steps_with_tool"] = int(_STATE["steps_with_tool"]) + 1
                    step_used_tool = True
                # Flush here rather than only at exit: an arm can be killed by the scheduler
                # (the runner sets --requeue), and a counter that only survives a clean exit
                # would lose precisely the runs that need explaining.
                _write_report()

                # Same conversation, extended. There is no assistant role available here
                # (`_openai_vision_messages` builds exactly one user turn), so the model's own
                # tool call is quoted back as text. This is the single-thread protocol the
                # transport already supports, and it keeps `api_planner.py` untouched.
                msgs.append({"type": "text", "text": f"Your previous output was:\n{out}"})
                msgs.append({"type": "text", "text": result.text})
                for idx in result.frames:
                    img = planner.frame_store_main.get(int(idx))
                    if img is not None:
                        msgs.append({"type": "image", "image": img})
                        wrist = planner.frame_store_wrist.get(int(idx))
                        if wrist is not None and getattr(planner, "use_wrist", False):
                            msgs.append({"type": "image", "image": wrist})

            return orig(system_prompt=system_prompt, user_content=msgs, **base_kw)
        except Exception as exc:  # noqa: BLE001
            _STATE["loop_errors"] = int(_STATE["loop_errors"]) + 1
            _STATE["errors"].append(f"tool loop: {exc!r}")
            sys.stderr.write(f"[memexp] tool loop failed, falling through: {exc!r}\n")
            # Fall through to a plain call rather than propagating: a broken memory layer must
            # degrade the arm to its baseline, not abort a 26-task job.
            try:
                return orig(system_prompt=system_prompt, user_content=user_content, **kw)
            except Exception:  # noqa: BLE001
                return None

    infer_primitive_via_api._memexp_wrapped = True
    module.infer_primitive_via_api = infer_primitive_via_api
    _STATE["api_call_wrapped"] = True


# ---------------------------------------------------------------------------------------
# Hook 3: reset the bank when the task or episode changes
# ---------------------------------------------------------------------------------------
def _wrap_resets(module) -> None:
    cls = module.ApiMemoryPlanner
    for name in ("reset_episode", "set_task_info"):
        orig = getattr(cls, name, None)
        if orig is None or getattr(orig, "_memexp_wrapped", False):
            continue

        def make(orig_fn, tag):
            def wrapper(self, *a, **k):
                out = orig_fn(self, *a, **k)
                if pull_enabled():
                    _reset_ctx(tag)
                return out
            wrapper._memexp_wrapped = True
            return wrapper

        setattr(cls, name, make(orig, name))
        _STATE.setdefault("reset_hooks", []).append(name)


def _bind(module) -> None:
    try:
        _wrap_build_messages(module)
        _wrap_api_call(module)
        _wrap_resets(module)
        # Hook 4 lives on a DIFFERENT class (the harness controller, constructed by the evaluator)
        # so it is installed separately and its failure is recorded rather than raised: an arm that
        # cannot see the stage must run and be reported as unopened, not crash the episode.
        _wrap_harness_stage()
        _STATE["module_patched"] = True
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"].append(f"bind failed: {exc!r}")
        sys.stderr.write(f"[memexp] BIND FAILED: {exc!r}\n")
        os.environ["MEMEXP_PULL_BIND_FAILED"] = "1"
    _write_report()


def install() -> None:
    """Install a meta_path hook that post-processes `harness.api_vlm_planner` on import.

    Not a direct `import ... ; patch` because at sitecustomize time the harness package is not
    importable yet (its directory is not on sys.path until the evaluator sets it up). The hook
    fires whenever the module is finally imported, which is the same reason
    `seam4_bind.install()` is built this way.
    """
    if _STATE["installed_at"] is not None:
        return
    _STATE["installed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Registered before anything can fail: a process that installs the hooks and then dies still
    # reports how far it got, which is the difference between "never engaged" and "engaged and
    # then crashed".
    _register_exit_report()

    # Resolve the tool registry HERE, at install time, when the tools are meant to be used. The
    # point is the timing: the same failure discovered inside the tool loop is caught by
    # `_build_messages`'s blanket handler and the arm continues as the push-only control, so a
    # misconfigured `MEMEXP_TOOLS_MODULE` would be reported as "the Planner chose not to call any
    # tool" -- a finding about the model, drawn from a typo in an environment variable.
    if pull_enabled() and tools_enabled():
        try:
            load_registry()
        except ToolsModuleError:
            # Deliberately not re-raised: this module is imported by `sitecustomize` in EVERY
            # interpreter that inherits PYTHONPATH, and raising here would take down the VLA server
            # and the runner's probes with it. The failure is recorded where the census reads it.
            pass

    from importlib.abc import MetaPathFinder
    from importlib.machinery import PathFinder

    class _PostExecLoader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            return self._wrapped.create_module(spec)

        def exec_module(self, module):
            self._wrapped.exec_module(module)
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
    if not pull_enabled():
        # Nothing to do and nothing to disturb. Other arms never reach the wrappers anyway, but
        # not touching the module at all keeps their behaviour provably identical.
        return
    module = sys.modules.get(_MOD)
    if module is not None:
        _bind(module)


# ---------------------------------------------------------------------------------------
# Verification: real probes, not a reading of the config
# ---------------------------------------------------------------------------------------
def verify_binding() -> dict:
    """Prove the hooks are live and the substrate behaves, using executable checks.

    Every check here corresponds to a failure this project has already paid for. Reading a flag
    is not evidence: job 586700 ran 26 tasks believing a feature was on while its own counter
    stayed at zero, precisely because the question "is the flag set?" was asked from outside the
    process that acts on it.
    """
    out: dict = {"ok": True, "problems": [], "state": dict(_STATE)}
    try:
        import harness.api_vlm_planner as m
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["problems"].append(f"cannot import {_MOD}: {exc!r}")
        return out

    live = {
        "build_messages_wrapped": bool(
            getattr(m.ApiMemoryPlanner._build_messages, "_memexp_wrapped", False)
        ),
        "api_call_wrapped": bool(getattr(m.infer_primitive_via_api, "_memexp_wrapped", False)),
        "pull_enabled_now": pull_enabled(),
        "tools_enabled_now": tools_enabled(),
    }
    out["live"] = live
    if not live["build_messages_wrapped"]:
        out["ok"] = False
        out["problems"].append("_build_messages not wrapped")
    if not live["api_call_wrapped"]:
        out["ok"] = False
        out["problems"].append("infer_primitive_via_api not wrapped (tool loop unreachable)")

    # --- substrate behaviour, on a real instance ------------------------------------------
    try:
        from memexp_substrate import Substrate

        # The registry the ARM will actually load, through the same validated loader the loop
        # uses. A preflight that validates a registry other than the one the run imports is worse
        # than no preflight at all: it certifies behaviour the arm will never exhibit, and the
        # difference is invisible because both registries expose the same class name.
        registry = load_registry()
        MemoryTools = registry.MemoryTools
        parse_tool_call = registry.parse_tool_call
        out["tools_module"] = tools_module_name()
        out["tools_module_error"] = _STATE.get("tools_module_error")

        sub = Substrate()
        # A frame store with 600 frames, so "frame exists" is a real predicate and not always
        # false. Without this, every read-time resolution would return nothing and both the gain
        # checks below would pass vacuously.
        sub.frame_store = {i: object() for i in range(600)}
        # ASSERTION 3 (arity): two containers must be two facts, not one overwritten slot. This
        # is the t14 defect -- `object_status` held one binding and "respectively" was therefore
        # unrepresentable, capping that task at 60.0 against an 80.0 baseline.
        sub.note_action("open middle drawer", 100)
        sub.note_action("open bottom drawer", 200)
        facts_mid, _ = sub.query("contents(middle drawer)")
        facts_bot, _ = sub.query("contents(bottom drawer)")
        out["live"]["arity_two_addresses_distinct"] = (
            len(facts_mid) == 1 and len(facts_bot) == 1
            and facts_mid[0].address != facts_bot[0].address
        )
        if not out["live"]["arity_two_addresses_distinct"]:
            out["ok"] = False
            out["problems"].append(
                "arity: two containers collapsed to one address -- a two-binding task would be "
                "unrepresentable"
            )

        # ASSERTION 1 (reachability): the tool must be reachable through the real dispatch path,
        # not merely present in a registry.
        tools = MemoryTools(sub)
        res = tools.call("query_world", {"address": "contents(middle drawer)", "_on_context": set()})
        out["live"]["query_world_reachable"] = bool(res.meta.get("ok"))
        row = out["live"]["query_world_reachable"] and "NEEDS LOOKING" in res.text
        out["live"]["query_returns_observation_not_answer"] = bool(row)
        if not res.meta.get("ok"):
            out["ok"] = False
            out["problems"].append(f"query_world did not resolve a minted address: {res.text[:120]}")

        # ASSERTION 2 (net gain): a served frame that is already on context is not evidence.
        # The positive control is checked FIRST (below, at `gain_first_query_serves_frames`),
        # because "every served frame is off-context" is vacuously true when nothing is ever
        # served. A filter assertion without a positive control is how a read path reports
        # healthy while doing nothing, which is the failure this project has already paid for
        # three times.
        out["live"]["gain_first_query_serves_frames"] = len(res.frames) > 0
        on_ctx = set(range(100, 105))
        r2 = tools.call("query_world", {"address": "contents(middle drawer)", "_on_context": on_ctx})
        off = [i for i in r2.frames if i not in on_ctx]
        out["live"]["gain_filter_excludes_on_context"] = all(i not in on_ctx for i in r2.frames)
        out["live"]["gain_off_context_frames"] = len(off)
        out["live"]["gain_probe_first_query_frames"] = len(res.frames)
        out["live"]["serve_is_bounded"] = len(res.frames) <= SERVE_MAX
        out["live"]["serve_cap"] = SERVE_MAX
        if not out["live"]["gain_first_query_serves_frames"]:
            out["ok"] = False
            out["problems"].append(
                "no frame was served for a freshly minted address: the filter check below would "
                "pass vacuously (zero-gain retrieval, reported as healthy)"
            )
        if not out["live"]["serve_is_bounded"]:
            out["ok"] = False
            out["problems"].append(
                f"a single dereference attached {len(res.frames)} frames (cap {SERVE_MAX})"
            )
        if not out["live"]["gain_filter_excludes_on_context"]:
            out["ok"] = False
            out["problems"].append("a served frame was already on context (zero-gain retrieval)")

        # A frame that does not exist yet must NOT be served, and the tool must say why. Serving
        # a nonexistent frame is the defect found by G0b-1: it attaches nothing while looking
        # like a successful retrieval.
        fresh = Substrate()
        fresh.frame_store = {}
        fresh.note_action("open middle drawer", 100)
        fr, _ = fresh.query("contents(middle drawer)")
        early = fresh.serve(fr, on_context=set())
        out["live"]["unobserved_frames_not_served"] = early == []
        future = Substrate()
        future.frame_store = {i: object() for i in range(600)}
        future.note_action("open middle drawer", 100)
        ff, _ = future.query("contents(middle drawer)")
        later = future.serve(ff, on_context=set())
        out["live"]["frames_appear_when_they_exist"] = len(later) > 0
        if not out["live"]["unobserved_frames_not_served"]:
            out["ok"] = False
            out["problems"].append("a not-yet-observed frame was served (attaches nothing)")
        if not out["live"]["frames_appear_when_they_exist"]:
            out["ok"] = False
            out["problems"].append("frames never became servable once they existed")

        # ASSERTION: the registry's stated finishing form must name the SAME keys the planner's own
        # instruction names. This is the measured root cause of the previous arm's 2.84% call rate:
        # the tool block taught `{"primitive": ...}` while the planning instruction -- the LAST
        # thing the model reads -- demanded `current_primitive`, and the parser discards the
        # former (probe: `{"primitive": "x"}` -> `primitive=''`, with no exception, so the
        # prose-fallback branch is never taken and the step yields no action at all).
        #
        # Asserted UNCONDITIONALLY now that both registries are corrected. It was opt-in while the
        # defect was still present in the default registry, because failing the gate there would
        # have blocked a completed arm rather than repaired it. `CONTRACT_AWARE` remains as a
        # declaration the arm files and the census can read, and a registry that opts OUT is
        # itself now a failure.
        spec = tools.spec_text()
        if not getattr(tools, "CONTRACT_AWARE", False):
            out["ok"] = False
            out["problems"].append(
                f"registry {tools_module_name()!r} does not declare CONTRACT_AWARE, so its "
                f"finishing form is not held to the planner's output contract"
            )
        planner_src = ""
        try:
            with open(
                os.path.join(
                    os.environ.get("ROOT", "."),
                    "evaluation_benchmark/harness/api_vlm_planner.py",
                ),
                encoding="utf-8",
            ) as fh:
                planner_src = fh.read()
        except Exception:  # noqa: BLE001
            planner_src = ""
        found = re.search(r"exactly two fields:\s*([A-Za-z_]+)\s+and\s+([A-Za-z_]+)", planner_src)
        contract = [found.group(1), found.group(2)] if found else []
        out["live"]["planner_contract_keys"] = contract
        # A gate that cannot find its own input must fail, not pass. `[]` is falsy, so an unnoticed
        # planner rewrite would silently turn every assertion below into a no-op.
        if not contract:
            out["ok"] = False
            out["problems"].append(
                "could not read the planner's output contract from api_vlm_planner.py: the "
                "contract-consistency assertions below would pass vacuously"
            )
        shown = [k for k in contract if k in spec]
        out["live"]["spec_names_planner_keys"] = sorted(shown)
        out["live"]["spec_missing_planner_keys"] = sorted(set(contract) - set(shown))
        if contract and len(shown) != len(contract):
            out["ok"] = False
            out["problems"].append(
                f"the tool spec never names {sorted(set(contract) - set(shown))}, but the planner "
                f"requires exactly those keys -- the model is being asked to break its own contract"
            )
        # The forced-finish clamp is the LAST thing the model reads before it must answer, so a key
        # name that appears only there and nowhere else is not a harmless typo: it is the final
        # word it is given, and it arrives exactly when the model is most likely to comply.
        clamp = force_final_text()
        bad_clamp = [
            k for k in re.findall(r'"([A-Za-z_]+)"\s*:', clamp) if contract and k not in contract
        ]
        out["live"]["forbidden_keys_in_clamp"] = sorted(set(bad_clamp))
        if bad_clamp:
            out["ok"] = False
            out["problems"].append(
                f"the forced-finish message names {sorted(set(bad_clamp))}, which the planner's "
                f"contract ({contract}) does not accept"
            )
        # The parser is the authority, so it is asked directly rather than second-guessed: the
        # spec must teach a form that the real parser turns into a NON-EMPTY primitive. This is
        # the assertion that would have caught the original defect; comparing key names alone
        # would pass as long as the two strings happened to be written the same way.
        try:
            from harness.vlm_output_parser import parse_vlm_output

            # The FINISHING form specifically, identified by its key rather than by being the
            # first JSON-ish object in the spec. The first version took the first match and
            # therefore parsed a tool call's `{"query": "<...>"}`, failing on a spec that was in
            # fact correct.
            taught = ""
            for cand in re.findall(r"\{[^{}]*\}", spec):
                if contract and contract[0] in cand:
                    taught = cand
                    break
            # CONCRETISED before parsing, and compared against the sentinel rather than merely
            # tested for truthiness -- see `concretise_form`. Parsing the raw form returns the raw
            # text for BOTH key names, so a truthiness test passes on the defect itself.
            probed = parse_vlm_output(concretise_form(taught), 600)[0] if taught else ""
            out["live"]["taught_form"] = taught
            out["live"]["taught_form_parsed"] = probed
            out["live"]["taught_form_parses_to_primitive"] = probed == FINISH_SENTINEL
            if not taught:
                out["ok"] = False
                out["problems"].append(
                    f"the spec shows no finishing form naming {contract[0] if contract else '?'}; "
                    f"the model is never told how to end the planning step"
                )
            elif probed != FINISH_SENTINEL:
                out["ok"] = False
                out["problems"].append(
                    f"the finishing form the spec teaches ({taught!r}) parses to {probed!r}, not "
                    f"the value of {contract[0] if contract else '?'} -- the model would comply "
                    f"and produce no action"
                )
        except Exception as exc:  # noqa: BLE001
            out["live"]["taught_form_parser_error"] = repr(exc)

        # ASSERTION: `search_memory` must reach a record WITHOUT the exact address. This is the
        # second measured cause -- `query_world` needs an address the model must already know, which
        # inverts retrieval. Exercised on the real dispatch path, not by inspecting the registry: a
        # tool present in a dict and a tool reachable by the model are different facts.
        if "search_memory" in getattr(tools, "tools", {}):
            sres = tools.call("search_memory", {"query": "drawer", "_on_context": set()})
            out["live"]["search_reaches_without_exact_address"] = bool(sres.meta.get("ok"))
            out["live"]["search_served_frames"] = len(sres.frames)
            out["live"]["search_positive_control_frames"] = len(sres.frames) > 0
            if not out["live"]["search_reaches_without_exact_address"]:
                out["ok"] = False
                out["problems"].append(
                    "search_memory could not reach a minted record from ordinary words: the arm "
                    "would still require the exact address and would not test what it claims"
                )
            if not out["live"]["search_positive_control_frames"]:
                out["ok"] = False
                out["problems"].append(
                    "search_memory resolved a record but served no frames: matching without "
                    "evidence is the zero-gain retrieval this project has already paid for"
                )

        # And the MESSAGE must match the situation. Telling the Planner "every frame is already
        # in your context" when the frames do not exist yet invites it to conclude the container
        # is empty -- a hallucination manufactured by the tool, which is worse than returning
        # nothing at all.
        early_tools = MemoryTools(fresh)
        early_res = early_tools.call(
            "query_world", {"address": "contents(middle drawer)", "_on_context": set()}
        )
        late_tools = MemoryTools(future)
        on_ctx_all = set(future.candidates(ff[0]))
        late_res = late_tools.call(
            "query_world", {"address": "contents(middle drawer)", "_on_context": on_ctx_all}
        )
        out["live"]["early_query_says_not_yet"] = "has not been made yet" in early_res.text
        out["live"]["late_query_says_already_have"] = (
            late_res.frames == [] and "already in your context" in late_res.text
        )
        if not out["live"]["early_query_says_not_yet"]:
            out["ok"] = False
            out["problems"].append(
                "an early query did not say the observation has not happened yet (the Planner "
                f"could read this as 'empty'): {early_res.text[-160:]!r}"
            )
        if not out["live"]["late_query_says_already_have"]:
            out["ok"] = False
            out["problems"].append(
                "a fully-on-context query was not reported as already available"
            )

        # A miss must be reported as a miss. Silent empty success is how a read path looks
        # healthy while doing nothing.
        miss = tools.call("query_world", {"address": "contents(spaceship)", "_on_context": set()})
        out["live"]["miss_is_reported"] = not miss.meta.get("ok")
        if not out["live"]["miss_is_reported"]:
            out["ok"] = False
            out["problems"].append("an unmatched address returned success")

        # Tool-call parsing must not swallow a primitive: if it did, the arm would score zero for
        # a reason indistinguishable from "the Planner produced nothing".
        prim = '{"current_primitive": "open top drawer", "keyframe_positions": []}'
        out["live"]["primitive_not_read_as_tool"] = parse_tool_call(prim, set(tools.tools)) is None
        tc = parse_tool_call('{"tool": "list_unbound"}', set(tools.tools))
        out["live"]["tool_call_parsed"] = bool(tc and tc["tool"] == "list_unbound")
        flat = parse_tool_call(
            '{"tool": "query_world", "address": "contents(top drawer)"}', set(tools.tools)
        )
        out["live"]["flat_args_accepted"] = bool(flat and flat["args"].get("address"))
        if not out["live"]["primitive_not_read_as_tool"]:
            out["ok"] = False
            out["problems"].append("a primitive was misread as a tool call")
        if not out["live"]["tool_call_parsed"]:
            out["ok"] = False
            out["problems"].append("a valid tool call did not parse")
        if not out["live"]["flat_args_accepted"]:
            out["ok"] = False
            out["problems"].append("flattened tool arguments were rejected")

        # Repetition inside one plan step must be served from cache WITHOUT re-attaching frames,
        # so the same pixels cannot masquerade as new evidence.
        tools.reset_step_cache()
        a = tools.call("query_world", {"address": "contents(bottom drawer)", "_on_context": set()})
        b = tools.call("query_world", {"address": "contents(bottom drawer)", "_on_context": set()})
        out["live"]["repeat_is_cached"] = bool(b.meta.get("repeat")) and not b.frames
        if not out["live"]["repeat_is_cached"]:
            out["ok"] = False
            out["problems"].append("a repeated query re-served frames instead of caching")

        # The push must be capped and must never contain a resolved answer.
        push = sub.render_push(max_actions=2, max_unbound=1)
        out["live"]["push_is_capped"] = push.count("\n  - ") <= 3
        out["live"]["push_nonempty"] = bool(push)
        if not out["live"]["push_nonempty"]:
            out["ok"] = False
            out["problems"].append("the small push rendered empty")

        # Tool spec wording must stay neutral. PMH's prompts ended with "Default: tool none" and
        # the measured result was an agent-less read path.
        spec = tools.spec_text().lower()
        out["live"]["spec_is_neutral"] = not any(
            bad in spec for bad in ("default: tool none", "prefer none", "avoid habitually")
        )
        if not out["live"]["spec_is_neutral"]:
            out["ok"] = False
            out["problems"].append("tool spec contains a discouraging default")

        # The spec must not solicit ENUMERATION. `list_facts`/`list_unbound` return the address
        # inventory and the resolution states and never an observation, and the push block rendered
        # into every plan step already carries that inventory -- so advertising them spends a full
        # model round-trip asking for what the model was just handed. Job 595132 measured it: 47 of
        # 95 tool calls (49%) were those two, and they returned 0 of the 19 frames the arm served.
        # They stay registered (a call to one still answers), so this asserts the OFFER, not the
        # capability -- which is the thing that changed.
        out["live"]["spec_does_not_solicit_enumeration"] = not any(
            f'"{n}"' in spec for n in ("list_facts", "list_unbound")
        )
        if not out["live"]["spec_does_not_solicit_enumeration"]:
            out["ok"] = False
            out["problems"].append(
                "the tool spec advertises an enumeration tool, so tool calls are being spent on a "
                "listing the push already rendered"
            )

        # The budget the spec states must be the budget the loop enforces. The previous spec said
        # "as many times as you need" next to a fuse of 3, and `cap_hits=2` in job 595132 counted
        # the contradiction rather than the model.
        _stated, _enforced = stated_budget(spec), _max_rounds()
        out["live"]["spec_budget_matches_fuse"] = _stated == _enforced
        if not out["live"]["spec_budget_matches_fuse"]:
            out["ok"] = False
            out["problems"].append(
                f"the spec states '{_stated}' tool call(s) but the fuse allows {_enforced}"
            )

        # ASSERTION 5 (scoping): the loop must engage ONLY on the main planning call. There are
        # four `infer_primitive_via_api` call sites; wrapping the stage-boundary call or a
        # PMH/SEAM decision call would inject a plan-step's `on_context` into a prompt that has
        # no live window, and the resulting behaviour would be attributed to the wrong mechanism.
        main, other = [{"type": "text", "text": "main"}], [{"type": "text", "text": "other"}]
        import memexp_bind as _self

        saved = getattr(_LOCAL, "ctx", None)
        try:
            _LOCAL.ctx = _Ctx(None, sub, tools, set(), 1, main, 0)
            out["live"]["scope_engages_on_main_call"] = _self._should_engage(main) is not None
            out["live"]["scope_skips_other_call"] = _self._should_engage(other) is None
            _LOCAL.ctx = None
            out["live"]["scope_skips_without_context"] = _self._should_engage(main) is None
        finally:
            _LOCAL.ctx = saved
        for key, why in (
            ("scope_engages_on_main_call", "the loop did not engage on the main planning call"),
            ("scope_skips_other_call",
             "the loop would wrap a non-planning API call (stale on_context leakage)"),
            ("scope_skips_without_context", "the loop engaged with no context at all"),
        ):
            if not out["live"][key]:
                out["ok"] = False
                out["problems"].append(why)

        # ---- cumulative accounting ------------------------------------------------------
        # The totals a verdict rests on must be PROCESS-wide, not a snapshot of whichever episode
        # happened to be live at the last write. Measured on job 592860: `tool_calls = 5` was
        # correct while `by_tool = {}` was empty, because the tool breakdown was read from a
        # per-episode object and the last episode of a 26-task run has no tool calls in it. A run
        # that called twenty tools can report zero per-tool calls, which is the one number that
        # distinguishes the search tool from the dereference tool.
        #
        # Exercised as an INTERLEAVED sequence, not as two independent episodes: the defect only
        # appears when an episode WITH activity is followed by one without, which is the shape of a
        # real 26-task run.
        saved_registry = list(_CTX_REGISTRY)
        saved_ctx = getattr(_LOCAL, "ctx", None)
        saved_state = {k: dict(_STATE.get(k) or {}) for k in ("by_tool", "tools_cumulative")}
        try:
            _CTX_REGISTRY.clear()
            bus = Substrate()
            bus.frame_store = {i: object() for i in range(600)}
            bus.note_action("open middle drawer", 100)
            with_tools = MemoryTools(bus)
            for _ in range(3):
                with_tools.call(
                    "query_world", {"address": "contents(middle drawer)", "_on_context": set()}
                )
            busy = _Ctx(None, bus, with_tools, set(), 9, None, 900)
            _CTX_REGISTRY.append(busy)
            _absorb()
            mid = dict(_STATE.get("by_tool") or {})
            out["live"]["cum_by_tool_after_busy_episode"] = mid.get("query_world", 0)

            # A second episode with NO tool calls, which is what retires the first one.
            quiet_bus = Substrate()
            quiet_bus.frame_store = {i: object() for i in range(600)}
            busy.closed = True
            _absorb_one(busy)
            quiet = _Ctx(None, quiet_bus, MemoryTools(quiet_bus), set(), 9, None, 901)
            _CTX_REGISTRY.append(quiet)
            _absorb()
            after = dict(_STATE.get("by_tool") or {})
            out["live"]["cum_by_tool_after_quiet_episode"] = after.get("query_world", 0)
            out["live"]["cum_by_tool_survives_quiet_episode"] = (
                after.get("query_world", 0) == mid.get("query_world", 0) == 3
            )
            if not out["live"]["cum_by_tool_survives_quiet_episode"]:
                out["ok"] = False
                out["problems"].append(
                    "per-tool counts are not cumulative across episodes: a run with calls would "
                    "report an empty by_tool as soon as a later episode made none -- the exact "
                    "shape of a 26-task run, and the only number that distinguishes "
                    "`search_memory` from `query_world`"
                )
            # Idempotence, because the report is written after every step AND at exit. A second
            # absorption that added the same deltas again would inflate every total.
            _absorb()
            twice = dict(_STATE.get("by_tool") or {})
            out["live"]["cum_absorb_is_idempotent"] = twice == after
            if not out["live"]["cum_absorb_is_idempotent"]:
                out["ok"] = False
                out["problems"].append(
                    "absorbing twice changed the totals, so the report's numbers depend on how "
                    "many times it happened to be written"
                )
        finally:
            _CTX_REGISTRY.clear()
            _CTX_REGISTRY.extend(saved_registry)
            _LOCAL.ctx = saved_ctx
            for k, v in saved_state.items():
                _STATE[k] = v

        # The declared registry list must be non-empty for a reporting run, and a registry that
        # failed to load must be reported rather than left to look like a model that never asked.
        out["live"]["tools_module_declared_in_state"] = _STATE.get("tools_module")
        out["live"]["registry_broken"] = registry_is_broken()
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["problems"].append(f"substrate/tool probe raised: {exc!r}")

    if _STATE["errors"]:
        out["ok"] = False
        out["problems"].extend(str(e) for e in _STATE["errors"])
    return out


if __name__ == "__main__":  # pragma: no cover
    install()
    print(json.dumps(verify_binding(), indent=2, sort_keys=True))
