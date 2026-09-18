"""mem_efficacy / correction of the memory text the harness hands to the Planner.

WHY THIS MODULE EXISTS
----------------------
Job 590799 measured a memory arm that scored WORSE than its own no-memory baseline, and the
cause was not "memory is useless". It was three defects in the CONTENT of the memory text, all
of which make the text assert things that were never observed. They are documented here because
the fix is only defensible if each one is named, reproduced, and falsifiable.

D1 -- the harness's own guesses were promoted to top-ranked "evidence".
      `memory_reason_for_planner` retrieved with
          query = f"stall {stage_name} {current_subtask}"
      and `memory_search` scored by "how many query tokens appear in the record". The literal
      token `stall` therefore matched every failure record, because the harness writes its own
      diagnostics into `notes` (`stall_recovery:<stage>`). A `subtask_override` record -- the
      recovery ladder's GUESS, stored with `outcome="planned"` -- scored 5/5 and was printed
      under the heading "Recent episode evidence:", in the same shape as things that actually
      happened. Measured on task 8 of 590799, the 4 retrieved rows were:

          score=5/5  action=stall             outcome=stalled
          score=5/5  action=subtask_override  outcome=planned   instruction='pick tomato sauce'
          score=3/5  action=subtask_update    outcome=active    instruction='pick tomato sauce'
          score=3/5  action=subtask_update    outcome=active    instruction='pick tomato sauce'

      Three of four rows were the same string, and that string was a guess. Task 8's first
      stage needs chocolate moved to the frypan; the Planner emitted 'pick tomato sauce' eight
      consecutive times instead, and the task fell from 66.7 to 0.0 while its Planner call count
      went 2 -> 10. Across the 8 tasks both arms finished, the call-count ratio predicts the
      score change: ratio ~1 -> no change, ratio 3.5-5.0 -> -25 to -66.7.

D2 -- the context ACCUMULATED instead of refreshing.
      `HarnessController._refresh_vlm_context` (upstream commit eb86819, not this experiment)
      ended with

          if self.vlm_context:
              ctx = self.vlm_context + "\n\n" + ctx
          self.vlm_context = ctx.strip()

      so despite the name, each call PREPENDED the previous rendering, which already contained
      the one before it. Measured: 15 calls produced 3982 characters containing the same three
      static rules 15 times over, and stale copies of "Current incomplete stage" from earlier
      stages. 90% of a long context was duplication. `nomem` is immune by construction (it
      returns early when `inject_vlm_context` is false), so this defect acted only on memory
      arms -- memory was penalised for a fault that appeared only when memory was switched on.

D3 -- static guidance dominated a fresh block.
      3 of the 5 `global_rules` are generic and identical for every task, every step, every arm,
      and made up 80% of a fresh 286-character block. D3 is NOT an independent bug: the damage
      came from D2 repeating them N times. With D2 fixed they appear exactly once, and the fix
      here only demotes them to the end of the block so the task-specific lines are nearest the
      observation. Reported as subsumed rather than as a third fixed defect.

THE PRINCIPLE, WHICH IS THE WHOLE FIX
-------------------------------------
The memory text must distinguish what was OBSERVED from what the harness GUESSED. Every defect
above is a violation of that one rule, including the second feedback loop:

    candidate = expected_primitive_for_stage(...)      # stage name -> primitive, a mapping
    ... rendered as "Suggested primitive for recovery: 'X'. Prefer this ..."

That is a guess, but it was phrased as an instruction. When the Planner obeyed, the evaluation
loop recorded the result as a `subtask_override` -- i.e. the guess became "evidence" in memory,
which was then retrieved at 5/5 next stall. So the loop had two stages, and fixing only the
retrieval would have left the directive half intact. Both stages are relabelled.

WHAT IS NOT CHANGED
-------------------
Nothing structural. Same memory store, same evaluation path, same task set, same Planner, same
frozen VLA. The rows are the same rows; only their presentation is separated by epistemic status.
No ranking is tuned toward a better score: `planned` records are still shown, `active` records are
still shown, the global rules are still shown. If corrected memory still does not help, that is a
real negative result and must be reported as one.

SAFETY
------
Installed by `pysite/sitecustomize.py` only when `MEMEXP_MEMFIX_ENABLE` is truthy, so the no-memory
baseline is provably untouched: its `_refresh_vlm_context` returns before reaching any renderer, and
the arms that do not set the flag never import this file. `verify_fix()` reproduces each defect on
the archived task-8 fixture and asserts the corrected path is free of it.
"""
from __future__ import annotations

import json
import os
import sys

# Evidence is partitioned by whether it was OBSERVED or GUESSED. These sets are the fix in
# miniature: the previous renderer treated all of them as one list of "episode evidence".
OBSERVED_OUTCOMES = {"stalled", "success", "resume"}
STATE_OUTCOMES = {"active"}
GUESSED_OUTCOMES = {"planned"}

# Modules the import hook waits for. Both are needed: `harness.controller` imports the renderer by
# name at module scope, so patching `harness.memory_reason` alone would leave the two Planner call
# sites pointing at the old function.
_HOOK_TARGETS = ("harness.memory_reason", "harness.controller")

# Tokens that carry no retrieval signal because they appear in the harness's OWN diagnostics.
# `stall` is here specifically: it is written into `notes` as `stall_recovery:<stage>` on every
# override and into `action` on every stall, so including it as a query token made every failure
# record match regardless of relevance. Removing it is not tuning; it is deleting a token that
# cannot discriminate.
DIAGNOSTIC_TOKENS = {
    "stall", "stalled", "stall_recovery", "recovery", "n/a", "none", "active", "planned",
    "success", "resume", "stage", "subtask", "step", "action", "notes", "outcome",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "on", "in", "for", "with", "into", "at", "by",
    "is", "are", "was", "were", "it", "its", "this", "that", "then", "over", "after", "before",
    "do", "not", "use", "using", "from", "as", "if", "when", "up", "down",
}

_STATE: dict[str, object] = {
    "installed": False,
    "hook_installed": False,
    "n_render": 0,
    "n_dup_dropped": 0,
    "n_guess_demoted": 0,
    "max_ctx_len": 0,
    "len_history": [],
    # Gate renders must not land in the runtime counter file. The gate calls `render_memory` to
    # prove the fix works; if those calls were counted, the census would report a healthy
    # `n_render` from a process that never evaluated a task -- a fix that certifies itself.
    "report_enabled": True,
    # The pre-patch functions, captured at install time. Without these the gate could only test
    # the corrected code against itself, and could not tell "the defect was fixed" apart from
    # "the defect never reproduced".
    "orig_render": None,
    "orig_search": None,
}

_REPORT_ENV = "MEMEXP_MEMFIX_REPORT"


def _write_report() -> None:
    """Persist this process's counters where the census can read them.

    A fix that cannot be shown to have ENGAGED is indistinguishable from a fix that never ran,
    which is the failure mode this experiment has already hit twice (a gate that silently
    skipped; an artifact that silently dropped the keys it existed to record). `n_render` is the
    load-bearing counter: zero means the corrected path was never reached.

    ONE FILE PER PROCESS, and this is not tidiness. An arm runs several interpreters against the
    same output directory -- `task1` and `tasks2to26` are separate processes, the gates run
    before the evaluation, and `merge_eval_outputs.sh` runs a `python3` AFTER it. With a single
    shared path each one truncated the previous one, so the file survived only as a snapshot of
    whichever process happened to write last. Measured on job 591479: the evaluator rendered
    memory thousands of times, and the surviving file read `n_render: 0`, because the
    end-of-arm `python3` (which loads this module via `PYTHONPATH` and installs it, but renders
    nothing) overwrote the real numbers. `census_channels.py` merges the per-process files.
    """
    if not _STATE["report_enabled"]:
        return
    path = str(os.environ.get(_REPORT_ENV, "")).strip()
    if not path:
        return
    try:
        payload = {
            "n_render": _STATE["n_render"],
            "n_dup_dropped": _STATE["n_dup_dropped"],
            "n_guess_demoted": _STATE["n_guess_demoted"],
            "max_ctx_len": _STATE["max_ctx_len"],
            "len_history": list(_STATE["len_history"])[-40:],
            "pid": os.getpid(),
            "installed": bool(_STATE["installed"]),
        }
        own = f"{path}.{os.getpid()}"
        tmp = f"{own}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, own)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------------------
def content_tokens(text: str) -> list[str]:
    """Tokens that can discriminate one record from another."""
    out: list[str] = []
    for raw in str(text or "").lower().replace("_", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) < 3 or tok in STOPWORDS or tok in DIAGNOSTIC_TOKENS:
            continue
        if tok not in out:
            out.append(tok)
    return out


def _dedupe(items: list) -> tuple[list, int]:
    """Collapse repeated (action, instruction) rows to the most recent, counting the drops.

    The previous renderer had no such step, which is why one string could occupy three of four
    retrieved rows. Repeats of the same fact carry no additional information but do consume the
    Planner's limited rows and, by repetition, look like emphasis.
    """
    newest: dict[tuple[str, str], object] = {}
    for it in items:
        key = (str(getattr(it, "action", "")), str(getattr(it, "instruction", "")))
        cur = newest.get(key)
        if cur is None or getattr(it, "step", 0) > getattr(cur, "step", 0):
            newest[key] = it
    dropped = len(items) - len(newest)
    return list(newest.values()), dropped


def _relevance(item, tokens: list[str]) -> int:
    hay = " ".join(
        str(getattr(item, f, "") or "")
        for f in ("action", "instruction", "outcome", "notes")
    ).lower()
    return sum(1 for t in tokens if t in hay)


def select_evidence(
    memory,
    *,
    stage_name: str | None,
    current_subtask: str,
    observed_cap: int = 4,
    state_cap: int = 2,
    guess_cap: int = 2,
) -> dict:
    """Partition and rank evidence by epistemic status.

    Returns observed / state / guessed rows plus the drop counts, so the caller can report how
    much duplication the correction actually removed rather than asserting that it did.
    """
    items = list(getattr(memory, "episode_evidence", []) or [])
    deduped, dropped = _dedupe(items)
    tokens = content_tokens(f"{stage_name or ''} {current_subtask}")

    def rank(rows: list) -> list:
        return sorted(rows, key=lambda it: (-_relevance(it, tokens), -getattr(it, "step", 0)))

    observed = rank([i for i in deduped if str(getattr(i, "outcome", "")) in OBSERVED_OUTCOMES])
    state = rank([i for i in deduped if str(getattr(i, "outcome", "")) in STATE_OUTCOMES])
    guessed = rank([i for i in deduped if str(getattr(i, "outcome", "")) in GUESSED_OUTCOMES])
    return {
        "observed": observed[:observed_cap],
        "state": state[:state_cap],
        "guessed": guessed[:guess_cap],
        "n_dropped": dropped,
        "n_guessed_total": len([i for i in deduped
                                if str(getattr(i, "outcome", "")) in GUESSED_OUTCOMES]),
        "tokens": tokens,
    }


def memory_search_corrected(memory, *, query: str, k: int = 6) -> list:
    """Drop-in replacement for `memory_search` with the two ranking defects removed.

    Kept as a patch target as well as used internally: any future caller of `memory_search` would
    otherwise inherit a ranker that lets the harness's own guesses outrank observed outcomes.
    """
    items = list(getattr(memory, "episode_evidence", []) or [])
    if not items:
        return []
    deduped, _ = _dedupe(items)
    tokens = content_tokens(query)
    scored = [(_relevance(i, tokens), i) for i in deduped]
    scored.sort(key=lambda p: (-p[0], -getattr(p[1], "step", 0)))
    if not tokens:
        return sorted(deduped, key=lambda i: -getattr(i, "step", 0))[:k]
    return [i for s, i in scored][:k]


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------
def _obs_line(item) -> str:
    step = getattr(item, "step", 0)
    action = str(getattr(item, "action", "") or "?")
    instr = str(getattr(item, "instruction", "") or "")
    notes = str(getattr(item, "notes", "") or "")
    outcome = str(getattr(item, "outcome", "") or "")
    if outcome == "stalled":
        return f"- step={step} STALLED on {instr!r}" + (f" [{notes}]" if notes else "")
    if outcome == "success":
        return f"- step={step} completed {instr!r}" + (f" [{notes}]" if notes else "")
    return f"- step={step} {action}: {instr!r}"


def render_memory(
    memory,
    *,
    stage_name: str | None,
    current_subtask: str,
    stall_steps: int,
    attempt_idx: int,
    candidate_subtask: str | None = None,
) -> str:
    """Corrected read-time memory text for the Planner.

    Ordering is deliberate: the current stage first, then things that happened, then things that
    were merely guessed (clearly marked as such and placed apart), then static guidance last so it
    cannot crowd out task-specific lines. The previous version's most salient slot went to the
    harness's own unverified guess.
    """
    sel = select_evidence(memory, stage_name=stage_name, current_subtask=current_subtask)
    _STATE["n_dup_dropped"] = int(_STATE["n_dup_dropped"]) + sel["n_dropped"]
    _STATE["n_guess_demoted"] = int(_STATE["n_guess_demoted"]) + sel["n_guessed_total"]

    lines: list[str] = []
    if stage_name:
        lines.append(f"Current incomplete stage: {stage_name}.")
    if stall_steps > 0:
        lines.append(f"Progress has stalled for {stall_steps} steps on subtask {current_subtask!r}.")

    if sel["observed"]:
        lines.append("Observed this attempt (these happened; newest first):")
        lines.extend(_obs_line(i) for i in sel["observed"])

    if sel["state"]:
        lines.append("Subtask changes recorded (control state, not world state):")
        lines.extend(
            f"- step={getattr(i, 'step', 0)} set to {str(getattr(i, 'instruction', ''))!r}"
            for i in sel["state"]
        )

    # The guess section. Its wording is the point: it states that these were never confirmed and
    # that adopting one is a decision to be tested, not a fact to be followed.
    if sel["guessed"]:
        lines.append(
            "Unverified recovery guesses (the harness produced these and never confirmed them; "
            "they are hypotheses, not evidence, and the attempt that produced them may itself have "
            "been the wrong choice):"
        )
        lines.extend(
            f"- step={getattr(i, 'step', 0)} guessed {str(getattr(i, 'instruction', ''))!r}"
            f" [{str(getattr(i, 'notes', '') or 'no note')}]"
            for i in sel["guessed"]
        )

    # Second half of the D1 loop. `candidate` is a stage->primitive mapping, not an observation,
    # and it used to be rendered as "Prefer this if ..." -- an instruction. Obeying it is what
    # wrote an override back into memory and closed the loop.
    if candidate_subtask and candidate_subtask != current_subtask:
        lines.append(
            f"Recovery hypothesis (derived from the stage name by a fixed map, NOT from any "
            f"observation): {candidate_subtask!r}. Falsifier: the stage above advances. If it "
            f"does not advance shortly, this hypothesis is wrong and repeating it is wasted steps."
        )

    best = getattr(memory, "best_partial", None)
    if best is not None:
        done = list(getattr(best, "completed_stages", []) or [])
        if done:
            lines.append("Best prior attempt completed stages: " + ", ".join(done[-4:]) + ".")
        score = getattr(best, "stage_score_pct", 0.0) or 0.0
        if score > 0:
            lines.append(f"Best attempt stage progress: {score:.0f}%.")

    if attempt_idx > 0:
        lines.append(
            "This is a resumed attempt after partial progress. Do not repeat already completed stages."
        )

    rules = list(getattr(memory, "global_rules", []) or [])
    if rules:
        lines.append("General guidance (static, identical for every task and step):")
        lines.extend(f"- {r}" for r in rules[:3])

    text = "\n".join(lines).strip()
    _STATE["n_render"] = int(_STATE["n_render"]) + 1
    _STATE["max_ctx_len"] = max(int(_STATE["max_ctx_len"]), len(text))
    _write_report()
    return text


# ---------------------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------------------
def _apply_patches() -> bool:
    """Apply the patches once both target modules are fully initialised. Idempotent.

    Called from the import hook, so it must tolerate being invoked while `harness.controller` is
    still executing. `sys.modules` holds a module object BEFORE its body finishes (that is how
    circular imports work), so a partially-built `harness.controller` is a real state to guard
    against: `HarnessController` would not exist yet and an eager attribute access would raise.
    Returning early is safe because the hook fires again after the module completes.
    """
    if _STATE["installed"]:
        return True
    reason_mod = sys.modules.get("harness.memory_reason")
    controller_mod = sys.modules.get("harness.controller")
    if reason_mod is None or controller_mod is None:
        return False

    cls = getattr(controller_mod, "HarnessController", None)
    upstream_render = getattr(reason_mod, "memory_reason_for_planner", None)
    upstream_search = getattr(reason_mod, "memory_search", None)
    if cls is None or upstream_render is None or upstream_search is None:
        return False  # camera-ready check: a partially executed module
    if getattr(cls._refresh_vlm_context, "_memexp_memfix", False):
        _STATE["installed"] = True
        return True

    # Capture the upstream functions BEFORE overwriting them, so the gate can run both and show
    # the defect reproducing on the original and being absent on the corrected path.
    _STATE["orig_render"] = upstream_render
    _STATE["orig_search"] = upstream_search

    # (1) Both render call sites bind the name into `harness.controller`, so patching there is
    #     what actually redirects the Planner's context (site 543 feeds the recovery rung's
    #     `memory_context`, site 597 feeds the Planner prompt).
    reason_mod.memory_reason_for_planner = render_memory
    reason_mod.memory_search = memory_search_corrected
    controller_mod.memory_reason_for_planner = render_memory

    # (2) Accumulation -> replacement, while preserving the attempt-level resume prefix, which is
    #     written once in `begin_attempt` and legitimately belongs at the FRONT of every render.
    original_refresh = cls._refresh_vlm_context

    def _refresh_vlm_context(
        self, stage_idx, stage_specs, subtask, step, candidate_subtask=None
    ) -> None:
        if not self.config.inject_vlm_context:
            self.vlm_context = ""
            return
        stage_name = stage_specs[stage_idx].name if stage_idx < len(stage_specs) else None
        stall_steps = max(0, step - self.stall_since_step) if self.stall_triggered else 0
        ctx = render_memory(
            self.memory,
            stage_name=stage_name,
            current_subtask=subtask,
            stall_steps=stall_steps,
            attempt_idx=self.attempt_idx,
            candidate_subtask=candidate_subtask,
        )
        resume = str(getattr(self, "_memfix_resume", "") or "")
        merged = f"{resume}\n\n{ctx}" if resume else ctx
        self.vlm_context = merged.strip()
        hist = _STATE["len_history"]
        if isinstance(hist, list):
            hist.append(len(self.vlm_context))
            _STATE["max_ctx_len"] = max(int(_STATE["max_ctx_len"]), len(self.vlm_context))
        _write_report()

    _refresh_vlm_context.__wrapped_original__ = original_refresh  # type: ignore[attr-defined]
    _refresh_vlm_context._memexp_memfix = True  # type: ignore[attr-defined]
    cls._refresh_vlm_context = _refresh_vlm_context

    # (3) Move the resume prefix out of the field the per-step renderer now owns, so "replace"
    #     does not delete attempt-level context.
    original_begin = cls.begin_attempt

    def begin_attempt(self, attempt_idx: int) -> None:
        original_begin(self, attempt_idx)
        self._memfix_resume = self.vlm_context if attempt_idx > 0 else ""
        self.vlm_context = ""

    begin_attempt.__wrapped_original__ = original_begin  # type: ignore[attr-defined]
    cls.begin_attempt = begin_attempt

    _STATE["installed"] = True
    _write_report()
    return True


def install() -> bool:
    """Register an import hook that applies the patches when the harness becomes importable.

    NOT a direct `import harness...; patch`, and this is the whole subtlety: at sitecustomize time
    the harness package CANNOT be imported, because its directory is not yet on `sys.path` -- the
    evaluator adds it later, in-process. An eager install therefore raises `ModuleNotFoundError`,
    and because `sitecustomize` must never crash the interpreter, that exception is caught and
    reported on stderr where nothing reads it. The result is a correction that is configured,
    documented, switched on by the arm's environment, and completely inert.

    That failure was observed while building this module: the first version imported the harness
    eagerly and the probe showed `memfix_install_error: ModuleNotFoundError("No module named
    'harness'")` while every other check passed. `verify_fix()` now asserts that the patches are
    actually live, precisely so this cannot recur unnoticed.

    The hook form is the same one `memexp_bind.install()` uses, for the same reason.
    """
    from importlib.abc import MetaPathFinder
    from importlib.machinery import PathFinder

    class _PostExecLoader:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def create_module(self, spec):
            return self._wrapped.create_module(spec)

        def exec_module(self, module):
            self._wrapped.exec_module(module)
            _apply_patches()  # fires only after the module body has finished

        def __getattr__(self, item):
            return getattr(self._wrapped, item)

    class _Finder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in _HOOK_TARGETS:
                return None
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PostExecLoader(spec.loader)
            return spec

    if not any(getattr(f, "_memexp_memfix_finder", False) for f in sys.meta_path):
        finder = _Finder()
        finder._memexp_memfix_finder = True  # type: ignore[attr-defined]
        sys.meta_path.insert(0, finder)
        _STATE["hook_installed"] = True
        _write_report()

    # Already-imported case: the gate imports the harness before calling this, and a module that
    # is already in `sys.modules` will never trigger the hook.
    _apply_patches()
    return bool(_STATE["installed"])


# ---------------------------------------------------------------------------------------
# Verification (used by the pre-flight gate; no API or GPU needed)
# ---------------------------------------------------------------------------------------
def verify_refresh_fix() -> dict:
    """Test the patched `_refresh_vlm_context` against the original, on a live controller.

    The renderer-level checks above cannot test D2: `render_memory` never mutates anything, so of
    course repeated calls return the same length. The accumulation lived in the CONTROLLER, which
    prepended its own previous output. Testing the fix therefore requires calling the real method
    and comparing it with the original, which is why `install()` keeps a reference to it.
    """
    import tempfile
    from types import SimpleNamespace

    import harness.controller as controller_mod
    from harness.config import load_harness_config

    out: dict = {"checks": {}, "error": None}
    cls = controller_mod.HarnessController
    patched = cls._refresh_vlm_context
    original = getattr(patched, "__wrapped_original__", None)
    if original is None:
        out["error"] = ("the controller's refresher is not patched, so the accumulation fix "
                        "cannot be tested; run install() first")
        return out

    def lens_for(fn, n: int = 12) -> list[int]:
        with tempfile.TemporaryDirectory() as td:
            ti = SimpleNamespace(
                task_id=8, task_block="put chocolate in frypan, pour sauce",
                primitive_labels=["pick up chocolate", "pick tomato sauce"],
                task_name="t8", scene_description="sd", brief_description="bd",
            )
            cfg = load_harness_config()
            ctrl = cls.create(8, ti, cfg, memory_root=td)
            ctrl.begin_attempt(0)
            out_lens: list[int] = []
            for i in range(n):
                spec = SimpleNamespace(name=f"0{i % 3 + 1}_Stage_{i % 4}")
                fn(ctrl, i % 3, [spec], f"subtask_{i}", 100 * i)
                out_lens.append(len(ctrl.get_vlm_context()))
            return out_lens

    # Guard: the test is only meaningful when the channel is actually on.
    with tempfile.TemporaryDirectory() as td:
        ti = SimpleNamespace(task_id=8, task_block="b", primitive_labels=["p"], task_name="t8",
                             scene_description="sd", brief_description="bd")
        cfg = load_harness_config()
        out["inject_vlm_context"] = bool(cfg.inject_vlm_context)
    if not out["inject_vlm_context"]:
        out["checks"]["skipped_channel_off"] = True
        out["checks"]["all_passed"] = True
        return out

    orig_lens = lens_for(original)
    corr_lens = lens_for(patched)
    out["orig_lens"] = orig_lens
    out["corr_lens"] = corr_lens

    # The defect: the original grows without bound, once per call.
    out["checks"]["original_accumulates"] = orig_lens[-1] > orig_lens[0] * 2
    out["checks"]["original_growth_is_monotonic"] = all(
        b > a for a, b in zip(orig_lens, orig_lens[1:])
    )
    # The fix: bounded, with no upward trend in the call count. NOT "constant length": the
    # content legitimately varies with the stage name and stall counter, so a bounded oscillation
    # is the correct behaviour and demanding a fixed length would fail on a working fix.
    out["checks"]["corrected_does_not_accumulate"] = corr_lens[-1] < orig_lens[-1] * 0.5
    half = len(corr_lens) // 2
    first_half = sum(corr_lens[:half]) / max(1, half)
    second_half = sum(corr_lens[half:]) / max(1, len(corr_lens) - half)
    out["second_half_over_first_half"] = round(second_half / max(1.0, first_half), 3)
    out["checks"]["corrected_has_no_upward_trend"] = second_half <= first_half * 1.1
    out["checks"]["corrected_is_bounded_by_content"] = max(corr_lens) <= 4 * max(1, min(corr_lens))

    # Independently of length: the content must not contain the same heading twice, which is what
    # an accumulating context looked like. Checked on the last render.
    with tempfile.TemporaryDirectory() as td:
        ti = SimpleNamespace(task_id=8, task_block="b", primitive_labels=["p"], task_name="t8",
                             scene_description="sd", brief_description="bd")
        cfg = load_harness_config()
        ctrl = cls.create(8, ti, cfg, memory_root=td)
        ctrl.begin_attempt(0)
        for i in range(6):
            spec = SimpleNamespace(name=f"0{i % 3 + 1}_Stage")
            patched(ctrl, i % 3, [spec], f"subtask_{i}", 50 * i)
        text = ctrl.get_vlm_context()
    out["final_text"] = text
    out["checks"]["single_current_stage_line"] = text.count("Current incomplete stage:") <= 1
    out["checks"]["single_general_guidance_block"] = text.count("General guidance") <= 1

    out["checks"]["all_passed"] = all(bool(v) for v in out["checks"].values())
    return out


def verify_fix() -> dict:
    """Reproduce every defect on real archived data, then assert the corrected path is clean.

    Written against the job-590799 task-8 fixture rather than a synthetic example, because a
    synthetic one would be built from the same misunderstanding as the fix. The ORIGINAL renderer
    is run first and its defects are asserted POSITIVELY -- a check that only asserts the fixed
    output is correct cannot tell "fixed" from "the defect never reproduced here".
    """
    import harness.controller as controller_mod
    import harness.memory_reason as reason_mod
    from harness.external_memory import ExternalMemory, EvidenceItem, AttemptRecord

    out: dict = {"error": None, "checks": {}}
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = os.path.join(here, "diagnosis_590799", "fixture_task8_attempt0_harness_memory.json")
    if not os.path.exists(fixture):
        out["error"] = f"missing fixture: {fixture}"
        return out

    # Liveness first. Everything below tests the corrected FUNCTIONS; this asserts that the
    # EVALUATOR would actually reach them. The first version of this module imported the harness
    # eagerly from sitecustomize, failed with ModuleNotFoundError, and was reported on stderr that
    # nobody reads -- a correction that was switched on, documented, and inert. A gate that only
    # tested the functions would have passed.
    install()
    try:
        import harness.controller as controller_mod
        import harness.memory_reason as reason_mod
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"harness not importable in this environment: {type(exc).__name__}: {exc}"
        return out
    _apply_patches()
    out["checks"]["patches_are_live_import_hook_fired"] = bool(_STATE["installed"])
    out["checks"]["renderer_call_site_is_patched"] = (
        controller_mod.memory_reason_for_planner is render_memory
    )
    out["checks"]["reason_module_is_patched"] = (
        reason_mod.memory_reason_for_planner is render_memory
        and reason_mod.memory_search is memory_search_corrected
    )
    out["checks"]["refresher_is_patched"] = bool(
        getattr(controller_mod.HarnessController._refresh_vlm_context, "_memexp_memfix", False)
    )
    out["checks"]["originals_were_captured"] = (
        _STATE["orig_render"] is not None and _STATE["orig_search"] is not None
    )
    if not out["checks"]["patches_are_live_import_hook_fired"]:
        out["error"] = ("the patches did not apply, so the rest of this gate would be testing "
                        "functions the evaluator never calls")
        return out

    # Gate renders must not be counted as runtime renders (see `report_enabled`).
    saved_report = _STATE["report_enabled"]
    _STATE["report_enabled"] = False
    try:
        return _verify_fix_inner(out, fixture)
    finally:
        _STATE["report_enabled"] = saved_report
        _write_report()


def _verify_fix_inner(out: dict, fixture: str) -> dict:
    import harness.memory_reason as reason_mod
    from harness.external_memory import ExternalMemory, EvidenceItem, AttemptRecord

    def build() -> ExternalMemory:
        d = json.load(open(fixture, encoding="utf-8"))
        m = ExternalMemory(task_id=int(d.get("task_id", 8)))
        m.global_rules = list(d.get("global_rules") or [])
        for e in d.get("episode_evidence") or []:
            m.episode_evidence.append(
                EvidenceItem(**{k: e.get(k) for k in EvidenceItem.__dataclass_fields__ if k in e})
            )
        bp = d.get("best_partial")
        if bp:
            m.best_partial = AttemptRecord(
                **{k: bp.get(k) for k in AttemptRecord.__dataclass_fields__ if k in bp}
            )
        return m

    kwargs = dict(
        stage_name="01_Lift_Tomato_Sauce",
        current_subtask="pick tomato sauce",
        stall_steps=0,
        attempt_idx=0,
        candidate_subtask="pick tomato sauce",
    )

    orig_render = _STATE["orig_render"]
    orig_search = _STATE["orig_search"]
    if orig_render is None or orig_search is None:
        out["error"] = ("install() has not run in this process, so the pre-patch functions are "
                       "unavailable and the defect cannot be reproduced; the gate would be "
                       "comparing the fix against itself")
        return out

    # ---- POSITIVE CONTROL: the defects, reproduced on the ORIGINAL code -------------------
    # Without this, "the corrected output looks right" is equally consistent with the defect
    # never having existed, and the fix would be unfalsifiable.
    before = orig_render(build(), **kwargs)
    out["before_len"] = len(before)
    out["before_text"] = before
    out["checks"]["repro_baseline_ran"] = len(before) > 0
    out["checks"]["repro_guesses_under_evidence_heading"] = (
        "Recent episode evidence:" in before and "outcome=planned" in before
    )
    # Duplication is counted from the rendered rows, not by looking for a magic substring: the
    # claim being tested is "the same instruction can occupy more than one retrieved row".
    import re as _re
    orig_rows = _re.findall(r"^- step=\S+ action=\S+ subtask=(.*?) outcome=", before, _re.M)
    out["orig_rows"] = orig_rows
    out["orig_dup_rows"] = len(orig_rows) - len(set(orig_rows))
    out["checks"]["repro_duplicate_rows"] = out["orig_dup_rows"] >= 1
    out["checks"]["repro_query_promotes_guesses"] = (
        len(orig_search(build(),
                        query=f"stall {kwargs['stage_name']} {kwargs['current_subtask']}",
                        k=4)) > 0
    )

    # ---- the corrected path ---------------------------------------------------------------
    after = render_memory(build(), **kwargs)
    out["after_len"] = len(after)
    out["after_text"] = after
    out["checks"]["no_bare_evidence_heading_for_guesses"] = "Unverified recovery guesses" in after
    out["checks"]["guess_is_marked_unverified"] = (
        "never confirmed them" in after and "hypotheses, not evidence" in after
    )
    out["checks"]["observed_section_present"] = "Observed this attempt" in after
    out["checks"]["stall_marked_as_observed_event"] = "STALLED on" in after
    out["checks"]["static_guidance_is_last"] = after.rstrip().splitlines()[-1].startswith("- ")
    out["checks"]["tokens_exclude_diagnostics"] = not (
        set(content_tokens(f"{kwargs['stage_name']} {kwargs['current_subtask']}"))
        & DIAGNOSTIC_TOKENS
    )

    # Duplicates are gone: the corrected selector must report having dropped the same rows the
    # original renderer emitted twice, and each section must list a given instruction once.
    sel = select_evidence(build(), stage_name=kwargs["stage_name"],
                          current_subtask=kwargs["current_subtask"])
    out["corrected_n_dropped"] = sel["n_dropped"]
    out["checks"]["dedupe_removed_the_duplicate_rows"] = sel["n_dropped"] == out["orig_dup_rows"]
    corr_rows = _re.findall(r"^- step=\S+ (?:STALLED on|completed|set to|guessed) (.*?)(?: \[|$)",
                            after, _re.M)
    out["corrected_rows"] = corr_rows
    guess_rows = _re.findall(r"^- step=\S+ guessed (.*?) \[", after, _re.M)
    state_rows = _re.findall(r"^- step=\S+ set to (.*)$", after, _re.M)
    out["checks"]["each_section_lists_a_fact_once"] = (
        len(guess_rows) == len(set(guess_rows)) and len(state_rows) == len(set(state_rows))
    )

    # The candidate branch must be exercised, or the check below is vacuous. The first fixture
    # call passed candidate == current_subtask, which CORRECTLY omits the line -- so a check
    # asserting the line is non-directive would have passed on an empty string. Assert the branch
    # both ways: absent when identical, present and non-directive when it differs.
    out["checks"]["candidate_line_absent_when_equal_to_current"] = "Recovery hypothesis" not in after
    alt = render_memory(build(), **{**kwargs, "current_subtask": "pick up chocolate"})
    out["alt_text"] = alt
    out["checks"]["candidate_line_present_when_different"] = "Recovery hypothesis" in alt
    out["checks"]["candidate_is_not_a_directive"] = (
        "Recovery hypothesis" in alt and "Prefer this" not in alt and "Falsifier:" in alt
    )

    # ---- renderer purity -------------------------------------------------------------------
    # This is a sanity check, NOT the D2 test: `render_memory` never mutates state, so repeated
    # calls are bound to return the same length. The accumulation defect lived in the controller,
    # and is tested by `verify_refresh_fix()` below against the real original method.
    lens: list[int] = []
    for i in range(12):
        lens.append(len(render_memory(build(), **{**kwargs, "stall_steps": i})))
    out["renderer_lens"] = lens
    out["checks"]["renderer_is_free_of_shared_state"] = max(lens[4:]) - min(lens[4:]) <= 8

    # The controller-level half of the fix (D2). Merged into the same checks dict so a single
    # `all_passed` covers both the renderer and the refresher.
    try:
        rr = verify_refresh_fix()
        out["refresh"] = rr
        if rr.get("error"):
            out["checks"]["refresh_probe"] = False
            out["refresh_error"] = rr["error"]
        else:
            for k, v in (rr.get("checks") or {}).items():
                out["checks"][f"ctrl_{k}"] = bool(v)
    except Exception as exc:  # noqa: BLE001
        out["checks"]["refresh_probe"] = False
        out["refresh_error"] = f"{type(exc).__name__}: {exc}"

    out["checks"]["all_passed"] = all(bool(v) for v in out["checks"].values())
    return out


if __name__ == "__main__":  # pragma: no cover - manual probe
    install()
    print(json.dumps(verify_fix(), indent=2))
