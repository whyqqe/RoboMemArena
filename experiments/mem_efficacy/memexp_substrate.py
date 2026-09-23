"""mem_efficacy / the deterministic memory substrate for the PULL arm.

WHAT THIS IS
------------
A bank of facts the robot has actually established, plus the set of addresses it has observed
but not yet resolved. It is written by the harness's own deterministic signals (the primitives
the Planner emitted, and the stages that advanced) -- never by asking a model what it thinks
happened. The PULL arm pushes a small summary of it every step and lets the Planner dereference
individual addresses on demand.

WHY A RELATIONSHIP TABLE AND NOT A KEY-VALUE STORE
--------------------------------------------------
This is a write-side theorem, and it has already been paid for once in this repository. PMH's
Task State held `object_status` as a SINGLE key, so it could store one binding. The question
"which of these two drawers held what, respectively" was therefore *unrepresentable*, and t14
sat at 60.0 against a baseline of 80.0 (see `harness/pmh_memory.py:384-388`). No retrieval
strategy can recover a fact the schema cannot hold.

The fix is that an address is a PAIR, not a name:

    contents(middle drawer) -> a row
    contents(bottom drawer) -> a different row

so two containers are two facts, not one overwritten slot. `_address` below is where that
arity is enforced, and `validate_arm.py` asserts it with a two-binding collapse test.

WHY THE VALUE IS USUALLY EMPTY, AND THAT IS THE POINT
-----------------------------------------------------
For the questions this benchmark actually asks (which drawer is empty, where did object X go),
the answer is only obtainable by LOOKING. So this substrate does not fabricate a text answer.
It records WHERE to look -- a step range around the observation -- and reports the address as
UNRESOLVED. The Planner is a VLM: it resolves the address by pulling the frames and reading
them. That keeps the write path deterministic and model-free, and it puts the perceptual
judgement where it belongs.

WHY FRAMES ARE RESOLVED AT READ TIME, NOT AT WRITE TIME
-------------------------------------------------------
A deliberate correction, found by `selftest_pull.py` G0b-1 rather than in a 26-task run. The
first version computed the frame list when the fact was created and stored it. But a fact is
created at the moment the primitive is PLANNED, and the frames that show its result -- the drawer
opened, the contents visible -- do not exist yet: the robot has not executed the primitive. The
stored list therefore named frames that were never in `frame_store_main`, every lookup attached
zero images, and the arm would have reported a healthy read path with an empty one. That is the
third time this project has met the same shape of defect.

So a `Fact` records only the STEP at which it was minted, and the concrete frames are computed
when the Planner asks -- filtered to the frames that exist by then, and to those not already on
context. A neighbourhood is also more correct than a single frame: the container is still
travelling at the first frame, so the evidence is the transition, and `MEMEXP_LOOKAHEAD` bounds
how much of it is offered.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# Container vocabulary, imported rather than re-declared. `redact.py` is the benchmark's single
# source of truth for what counts as a container, and a second copy here would eventually
# disagree with it -- which is exactly the drift `redact.py`'s own header was written to prevent.
try:
    from harness.redact import CONTAINER_NOUNS, QUALIFIERS
except Exception:  # pragma: no cover - only when imported outside the harness environment
    CONTAINER_NOUNS = {
        "drawer", "drawers", "basket", "baskets", "cabinet", "microwave", "drainer",
        "bowl", "bowls", "shelf", "tray", "oven", "fridge", "refrigerator", "sink",
        "plate", "bin", "box", "pot", "carton", "cup", "mug", "container", "rack",
    }
    QUALIFIERS = {
        "top", "middle", "bottom", "upper", "lower", "left", "right", "front", "back",
        "side", "first", "second", "third", "fourth", "inner", "outer", "near", "far",
    }

# Verbs that create an observation worth re-checking later. Opening a container is the canonical
# one: the contents become visible and are then occluded again by the close.
_OPEN_VERBS = {"open", "pull", "slide"}
_CLOSE_VERBS = {"close", "shut", "push"}

# How far past the mint step the observation is sampled. `REPLAN_STEPS` is 10, so a primitive
# planned now is executed over roughly the next 10 steps and the container is settled near the
# end of that. A window of 1 would only ever offer the frame in which the drawer starts moving.
LOOKAHEAD = int(os.environ.get("MEMEXP_LOOKAHEAD", "8"))

# The most images one dereference may attach. Bounded because every attached image costs tokens
# on every subsequent call in the tool loop, and an unbounded window turns one useful
# observation into a conversation the model cannot afford.
SERVE_MAX = int(os.environ.get("MEMEXP_SERVE_MAX", "4"))

# The mint step given to a task-spec address before its stage activates. It is deliberately far
# in the future rather than 0 so that `candidates()` names frames that cannot exist yet and
# `available_frames()` returns nothing: a seeded address must be VISIBLE (it is in the push, so
# the Planner knows the address exists) without being READABLE (its frames would be the untouched
# initial scene). A step counter that is still below this after an episode is a bug; nothing
# compares against it numerically except the two predicates above.
_NOT_YET = 10 ** 9


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", str(text or "").lower()) if t]


def container_phrases(text: str) -> set[str]:
    """Extract every container mention, qualified or bare, as a canonical key.

    Deliberately laxer than `redact.container_phrases`, which requires a qualifier. That
    strictness is right for withholding (only "top drawer" is an answer key) but wrong here: an
    address must exist for a bare "drawer" too, because a primitive like "put the butter in the
    drawer" still creates an observation. Qualified forms stay distinct, so "middle drawer" and
    "top drawer" do not collapse into one address.
    """
    toks = _tokens(text)
    out: set[str] = set()
    for i, tok in enumerate(toks):
        if tok not in CONTAINER_NOUNS:
            continue
        noun = tok[:-1] if tok.endswith("s") and tok[:-1] in CONTAINER_NOUNS else tok
        j, k = i, i - 1
        while k >= 0 and toks[k] in QUALIFIERS and (i - k) <= 2:
            j = k
            k -= 1
        out.add(" ".join(toks[j:i] + [noun]) if j < i else noun)
    return out


def _address(container: str) -> str:
    """The canonical address for 'what is inside this container'.

    The parenthesised form is not decoration: it is what stops two containers from occupying one
    slot. See the module header for the measured cost of getting this wrong.
    """
    return f"contents({container})"


def _stage_address(stage_name: str) -> str:
    """The canonical address for 'has this stage of the task been achieved'.

    Separate from `_address` because it is a different KIND of fact: `contents(x)` is answered by
    looking, `stage(x)` is answered by the harness's own predicate. Keeping them in one namespace
    would let a `query_world("stage(...)")` be answered with frames, which is exactly the
    confusion between "where to look" and "what is true" this module is built to avoid.
    """
    return f"stage({stage_name})"


@dataclass
class Fact:
    """One thing the robot established, and the step range in which it can be looked at."""
    address: str
    value: str            # "" when the answer needs looking at, which is the normal case
    mint_step: int        # the step at which this was written; frames derive from it at read time
    source: str = "action"   # "action" | "stage" | "spec" | "verified"
    resolved: bool = False   # True once the Planner has pulled it and seen the frames
    spec_stage: str = ""     # for `source="spec"`: the stage that will open this address


class Substrate:
    """The fact bank. One instance per episode; `reset()` between attempts."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.by_address: dict[str, list[int]] = {}
        # Set by the binding to the Planner's live frame store. Frames are resolved against it at
        # read time, which is the only moment the answer to "does this frame exist?" is knowable.
        self.frame_store: dict[int, Any] = {}
        # PrediMem-style preferred frame indices for the next serve. Empty means "use the temporal
        # window alone". See `set_retrieval_bank`.
        self.retrieval_bank: list[int] = []
        # Instrumentation. Every counter here exists because the alternative is a run whose
        # failure mode is indistinguishable from "memory did not help" -- the lesson from PMH's
        # `PMH_MAX_SEARCH_PER_PLAN=0`, where a branch was unreachable in code while the arm still
        # reported itself enabled.
        self.n_action_write = 0
        self.n_unbound_minted = 0
        self.n_query = 0
        self.n_query_hit = 0
        self.n_query_miss = 0
        self.n_frames_served_off_context = 0
        self.n_serve_empty_not_yet_observed = 0
        # Write-path-2 instrumentation. `n_spec_minted > 0` is the falsifier for "the task-spec
        # seed actually ran"; a run with it at 0 has a bank built the old way and its score is
        # not a statement about this write path. Same lesson as PMH's unreachable
        # `PMH_MAX_SEARCH_PER_PLAN=0` branch.
        self.n_spec_minted = 0
        self.n_stage_active = 0
        self.n_stage_verified = 0
        self._active_stage = ""
        self._verified_seen: set[str] = set()
        self._last_subtask = ""
        self._last_subtask_step = -1

    # ---- write path ----------------------------------------------------------------------
    def reset(self) -> None:
        self.facts.clear()
        self.by_address.clear()
        self.retrieval_bank = []
        self._active_stage = ""
        self._verified_seen.clear()
        self._last_subtask = ""
        self._last_subtask_step = -1

    def note_action(self, primitive: str, step: int) -> int:
        """Record a primitive the Planner emitted, and mint any addresses it opens up.

        Returns the number of NEW unresolved addresses minted. Called from the prompt builder,
        which runs once per plan step and by which point the previous step's primitive is known.
        Idempotent in the (primitive, step) pair so a retry of the same plan step cannot
        double-write.
        """
        text = str(primitive or "").strip()
        if not text:
            return 0
        if int(step) == self._last_subtask_step and text == self._last_subtask:
            return 0
        self._last_subtask = text
        self._last_subtask_step = int(step)

        tokens = _tokens(text)
        if not tokens:
            return 0
        verb = tokens[0]
        containers = container_phrases(text)
        minted = 0

        if verb in _OPEN_VERBS:
            for c in sorted(containers):
                addr = _address(c)
                if addr in self.by_address:
                    continue
                self._append(Fact(
                    address=addr,
                    value="",        # unresolved by construction: an answer needs eyes
                    mint_step=int(step),
                    source="action",
                ))
                minted += 1
                self.n_unbound_minted += 1
        elif verb in _CLOSE_VERBS:
            # Closing is recorded as an action but mints nothing: it HIDES evidence rather than
            # creating it, and minting here would ask the Planner to look at a shut drawer.
            pass
        else:
            # Every other primitive still advances the episode, so it is recorded as a fact about
            # what happened. This is what makes "what have I already done" answerable without the
            # Planner having to remember its own history.
            for c in sorted(containers):
                self._append(Fact(
                    address=f"touched({c})",
                    value=text,
                    mint_step=int(step),
                    source="action",
                ))
        self.n_action_write += 1
        return minted

    def note_stage(self, name: str, step: int) -> None:
        if not str(name or "").strip():
            return
        self._append(Fact(
            address="stage",
            value=str(name).strip(),
            mint_step=int(step),
            source="stage",
        ))

    # ---- write path 2: the TASK SPEC (deterministic, model-free) ---------------------------
    def seed_from_stages(self, stage_names: "list[str] | tuple[str, ...]", step: int) -> int:
        """Materialise the task's entity space from the task spec, at episode start.

        WHY THIS EXISTS -- and it is a diagnosis, not a convenience. The action write path above
        mints an address only for a container the PLANNER named. So on task 4, where the Planner
        in this project's traces emits `open top drawer` on 120/120 steps, the bank learns about
        exactly one container and the address the task actually needs -- "which of middle/bottom
        drawer still holds something" -- is never created. The Planner cannot pull a fact the
        schema never held; that is the same write-side theorem the module header records from
        PMH's single-key `object_status` (t14: 60.0 vs 80.0), one level up.

        The stage list is a PURE FUNCTION of the task id (`task2_26_reference_stage._task_specs`)
        and the harness already computes it to score stages. It names every container and object
        the task involves, in the task's own required order. Seeding from it is therefore the
        write-time inference step that a frozen general VLM cannot do for itself: the CODE knows
        the task skeleton, the model does not have to rediscover it.

        An address seeded here is NOT readable yet -- `mint_step` is set to `_NOT_YET`, so
        `candidates()` names frames that do not exist and `available_frames()` returns nothing.
        It becomes readable when its stage activates (`note_stage_active`), which is the step the
        robot actually works on it. Without that, a seed at step 0 would advertise eight frames
        of the untouched initial scene as "evidence" -- the exact defect `selftest_pull.py` G0b-1
        was written to catch for action-minted facts.

        Returns the number of new addresses minted.
        """
        minted = 0
        for name in list(stage_names or []):
            nm = str(name or "").strip()
            if not nm:
                continue
            addr = _stage_address(nm)
            if addr not in self.by_address:
                self._append(Fact(
                    address=addr, value="", mint_step=_NOT_YET,
                    source="spec", spec_stage=nm,
                ))
                minted += 1
            for c in sorted(container_phrases(nm)):
                a = _address(c)
                # Keep the EARLIEST stage that needs a container: the first moment the task
                # requires looking into it is when its evidence window should open. Re-binding
                # it to a later stage would close a window the Planner may still need.
                if a in self.by_address:
                    continue
                self._append(Fact(
                    address=a, value="", mint_step=_NOT_YET,
                    source="spec", spec_stage=nm,
                ))
                minted += 1
        self.n_spec_minted += minted
        return minted

    def note_stage_active(self, name: str, step: int) -> int:
        """Open the evidence window for every spec address belonging to the activating stage.

        Driven by the harness's own stage machine (`episodic_store.set_current_stage`), which
        advances only when a stage's predicate passed on the REAL environment. So the window is
        anchored to a verified transition rather than to the Planner's belief about where it is.

        Returns the number of addresses opened.
        """
        nm = str(name or "").strip()
        if not nm or nm == self._active_stage:
            return 0
        self._active_stage = nm
        self.n_stage_active += 1
        opened = 0
        for f in self.facts:
            if f.source == "spec" and f.spec_stage == nm and f.mint_step == _NOT_YET:
                f.mint_step = int(step)
                opened += 1
        return opened

    def note_verified_stage(self, name: str, step: int) -> None:
        """Record a stage completion as a RESOLVED fact, from the harness's ground-truth check.

        This is the one place the substrate is allowed to hold a text answer rather than an
        address, and the reason is that the answer is not perceptual: `spec.check_fn(env, state,
        start)` already established it against the environment. Recording it lets a later query
        ("did the middle drawer stage finish?") return text instead of sending the Planner back
        to pixels to re-derive something the harness already knows.
        """
        nm = str(name or "").strip()
        if not nm or nm in self._verified_seen:
            return
        self._verified_seen.add(nm)
        self.n_stage_verified += 1
        addr = _stage_address(nm)
        hit = False
        for i in self.by_address.get(addr, []):
            f = self.facts[i]
            f.value = f"{nm} completed (verified against the environment)"
            f.resolved = True
            if f.mint_step >= _NOT_YET:
                f.mint_step = int(step)
            hit = True
        if not hit:
            self._append(Fact(
                address=addr,
                value=f"{nm} completed (verified against the environment)",
                mint_step=int(step),
                source="stage",
                resolved=True,
                spec_stage=nm,
            ))

    def _append(self, fact: Fact) -> None:
        self.facts.append(fact)
        self.by_address.setdefault(fact.address, []).append(len(self.facts) - 1)

    # ---- read path -----------------------------------------------------------------------
    def unbound(self) -> list[Fact]:
        return [f for f in self.facts if not f.value and not f.resolved]

    def recent_actions(self, n: int) -> list[Fact]:
        acts = [f for f in self.facts if f.source in {"action", "stage"}]
        return acts[-n:]

    def candidates(self, fact: Fact) -> list[int]:
        """Every frame that could show this fact's result, whether or not it exists yet."""
        if fact.mint_step >= _NOT_YET:
            return []
        return list(range(fact.mint_step, fact.mint_step + LOOKAHEAD + 1))

    def set_retrieval_bank(self, indices: "list[int] | tuple[int, ...] | None") -> None:
        """Install a PrediMem-style preferred frame set for the next serve.

        WHY THIS EXISTS. Without it, `available_frames` can only pick from the temporal window
        around `mint_step` -- the LOOKAHEAD neighbours of the action. PrediMem's bank, by
        contrast, is built from nominations and stage/subtask anchors: frames that mark STATE
        CHANGES. When the pull arm builds that bank but does not push it (so the Planner must ask),
        the serve path has to prefer those frames; otherwise the bank is built and then ignored,
        and the pull returns the same local neighbourhood a no-memory look-back would have.

        Empty / None clears the preference and restores the pure temporal window -- which is what
        the push-only and offline probes need.
        """
        if not indices:
            self.retrieval_bank = []
            return
        try:
            self.retrieval_bank = sorted({int(i) for i in indices if int(i) >= 0})
        except Exception:
            self.retrieval_bank = []

    def available_frames(self, fact: Fact, on_context: set[int] | None = None) -> list[int]:
        """The frames worth attaching for this fact, right now.

        Four filters, and each one is a defect this project has already paid for:
          * the frame must EXIST -- see the module header; a frame that is not in the store
            attaches nothing while looking like a successful retrieval;
          * it must not be ON CONTEXT -- re-serving what the Planner can already see moved no
            information, which is what made job 570810's `picked=1` sit beside
            `n_gain_skipped=101`;
          * when a PrediMem retrieval bank is installed, prefer its frames that fall inside this
            fact's observation window -- those are the transition-marked frames, not the local
            neighbourhood of the mint step;
          * the result must be BOUNDED, so the token cost of the loop stays predictable.
        """
        excl = set(on_context or set())
        window = set(self.candidates(fact))
        bank = list(getattr(self, "retrieval_bank", []) or [])
        if bank:
            # PrediMem preference. If the bank has ANY frame inside this fact's observation
            # window, serve from that intersection (minus on-context). Do NOT fall back to the
            # temporal neighbourhood when the intersection is empty only because those frames
            # are already on context -- that fallback would re-serve less-informative neighbours
            # and hide the fact that the PrediMem frames were already consumed.
            bank_in_window = [
                i for i in bank if i in window and i in self.frame_store
            ]
            if bank_in_window:
                live = [i for i in bank_in_window if i not in excl]
            else:
                live = [
                    i for i in self.candidates(fact)
                    if i in self.frame_store and i not in excl
                ]
        else:
            live = [
                i for i in self.candidates(fact)
                if i in self.frame_store and i not in excl
            ]
        if not live:
            return []
        if len(live) <= SERVE_MAX:
            return live
        # Evenly spaced, always including both ends: the first frame shows the transition and
        # the last shows the settled state, which are the two moments that carry the contents.
        span = len(live) - 1
        pick = sorted({live[round(k * span / (SERVE_MAX - 1))] for k in range(SERVE_MAX)})
        return pick

    def query(self, address: str, on_context: set[int] | None = None) -> tuple[list[Fact], str]:
        """Dereference an address. Returns (facts, resolution_status).

        Exact address match first, then a token-overlap fallback so a Planner that phrases the
        address slightly differently still reaches the row. No embedding similarity is used, and
        that is deliberate: the question names a STATE ("which drawer is empty") while the record
        holds an ACTION ("opened middle drawer"), and those two share almost no vocabulary. The
        escape from that mismatch is to resolve through the address, not to rank by surface
        similarity -- ranking by similarity returns the most RECENT relevant row, which the
        Planner can already see, which is why raising PMH's retrieval budget 0 -> 1 -> 2 measured
        as a no-op every time.
        """
        self.n_query += 1
        q = str(address or "").strip()
        exact = self.by_address.get(q)
        if exact:
            self.n_query_hit += 1
            return [self.facts[i] for i in exact], "hit"

        qt = set(_tokens(q))
        qt -= {"contents", "the", "a", "an", "in", "of", "what", "is", "which", "was", "where"}
        if not qt:
            self.n_query_miss += 1
            return [], "miss"
        scored: list[tuple[int, Fact]] = []
        for f in self.facts:
            overlap = len(qt & set(_tokens(f.address)))
            if overlap:
                scored.append((overlap, f))
        if not scored:
            self.n_query_miss += 1
            return [], "miss"
        scored.sort(key=lambda x: (-x[0], -x[1].mint_step))
        best = scored[0][0]
        self.n_query_hit += 1
        return [f for s, f in scored if s == best], "fallback"

    def window_elapsed(self, fact: Fact) -> bool:
        """True when the observation window has closed -- no further frames will arrive.

        This is the public predicate the tool layer needs, because "may I serve more later?" is
        NOT the same question as "do I have frames right now?", and answering the second one in
        place of the first produces a wrong message. The tool's first version asked
        `available_frames(fact, set())` -- with an empty context, so it counted frames that were
        in fact on context -- and therefore told the Planner "every frame is already in your
        context" when the truth was "the frames do not exist yet". One of those messages invites
        the Planner to conclude the container is empty.

        Three cases qualify, and nothing else does:
          * the fact never had frames to look at (a `stage` record, say) -- the text IS the answer;
          * the window has fully elapsed, so returning none means they are all on context;
          * there is no frame store at all (a probe or an offline replay), so waiting is pointless.

        The middle case took two attempts to get right, and the wrong version is worth recording.
        It first asked "does ANY candidate frame exist?", which is true as soon as the FIRST frame
        of the window is captured -- and that frame is by construction still on context, so the
        answer was empty while the fact was marked settled. A Planner that asked one step after
        the action therefore got nothing AND lost the address. The window is the unit, not the
        frame: the question is whether the evidence has finished arriving.
        """
        cands = self.candidates(fact)
        if not cands:
            return True
        if not self.frame_store:
            return False
        return fact.mint_step + LOOKAHEAD <= max(self.frame_store)

    # Kept as a private alias: `serve` reads more naturally calling it this way, and the two must
    # never diverge -- they were separate predicates once and that is how the bug above survived.
    _answerable = window_elapsed

    def serve(self, facts: list[Fact], on_context: set[int] | None = None) -> list[int]:
        """Mark the pulled facts resolved and return their frames, minus what is on context.

        A fact is marked resolved ONLY when the dereference actually settled it. An initial
        version marked every queried fact resolved, which produced a defect worse than the one it
        replaced: a Planner that asked one step too early -- before the frames it wanted had been
        observed -- got an empty answer AND lost the address, because `unbound()` no longer
        returned it and the push stopped advertising it. The evidence stayed in the episode and
        became unreachable, so the failure was permanent for that attempt. An unanswerable query
        must leave the address exactly where it was, so it is advertised again the moment its
        frames exist. That recovery is asserted by `selftest_pull.py` G0b-6.
        """
        out: list[int] = []
        pending = 0
        for f in facts:
            frames = self.available_frames(f, on_context)
            if frames:
                for i in frames:
                    if i not in out:
                        out.append(i)
                f.resolved = True
            elif self._answerable(f):
                # Frames exist and are on context: the Planner has them, so this is answered.
                f.resolved = True
            else:
                # Not observed yet. Deliberately left UNRESOLVED so it can be re-advertised.
                pending += 1
        if pending:
            self.n_serve_empty_not_yet_observed += pending
        self.n_frames_served_off_context += len(out)
        return out

    # ---- the small push -------------------------------------------------------------------
    def render_push(self, max_actions: int = 4, max_unbound: int = 4,
                    on_context: set[int] | None = None) -> str:
        """Render the compact block that goes into every prompt.

        Capped on both axes and never containing an answer. The push exists so the Planner knows
        which addresses it COULD ask about -- i.e. it converts "discover that you are missing
        something" (unreliable: PMH asked a model to self-assess and got 85% 'none') into "read a
        list of open addresses" (a structural fact). The autonomy is in choosing and phrasing the
        query, not in having to guess that a gap exists.

        ONLY ADDRESSES THAT WILL ACTUALLY RETURN PIXELS ARE ADVERTISED. An address is minted at
        the step its primitive is PLANNED, and the frames showing the result do not exist until
        the robot has executed it -- so an address advertised immediately is an invitation to a
        round trip whose answer is "nothing yet". The census has a dedicated counter for exactly
        that outcome (`n_serve_empty_not_yet_observed`), and the first draft of this module
        produced it on every query. The address is still minted and still recorded; it simply
        appears in the push from the step where it becomes readable, which is a few steps later.
        """
        excl = set(on_context or set())
        ready, deferred, upcoming = [], [], []
        for f in self.unbound():
            if f.source == "spec" and f.mint_step == _NOT_YET:
                upcoming.append(f)
            elif self.available_frames(f, excl):
                ready.append(f)
            else:
                deferred.append(f)

        lines: list[str] = []
        acts = self.recent_actions(max_actions)
        if acts:
            lines.append("recent actions (addresses can be dereferenced with query_world):")
            for f in acts:
                lines.append(f"  - step {f.mint_step}: {f.value or f.address}")
        if upcoming:
            # The task-spec addresses. Listed so the Planner knows, without guessing, which
            # entities this task involves and in what order -- that is the whole point of seeding
            # from the spec. Explicitly labelled "not yet observed" so an unreadable address is
            # never mistaken for an empty container, which is the failure a bare omission (or a
            # premature "0 frames ready") would produce.
            lines.append(
                "task addresses NOT YET OBSERVED (from the task specification; each becomes "
                "readable when the task reaches its stage):"
            )
            for f in upcoming[:max_unbound]:
                lines.append(f"  - {f.address}   [stage {f.spec_stage}]")
            if len(upcoming) > max_unbound:
                lines.append(f"  - ... and {len(upcoming) - max_unbound} more (use list_unbound)")
        if ready:
            lines.append(
                "observed but NOT YET RESOLVED (the answer needs looking at; "
                "call query_world with the address):"
            )
            for f in ready[:max_unbound]:
                n = len(self.available_frames(f, excl))
                lines.append(f"  - {f.address}   [{n} frame(s) ready to look at]")
            if len(ready) > max_unbound:
                lines.append(f"  - ... and {len(ready) - max_unbound} more (use list_unbound)")
        if deferred:
            # Named, but explicitly as "not yet". A silent omission would be worse than either
            # alternative: the Planner would know the address had been observed (the action is
            # in the list above) and be unable to tell whether asking was pointless or broken.
            names = ", ".join(f.address for f in deferred[:3])
            lines.append(
                f"observed but not yet readable (no frames captured yet): {names}"
                f"{' ...' if len(deferred) > 3 else ''}"
                " -- these become readable a few steps after the action; ask then."
            )
        if not lines:
            return ""
        return "=== memory (pushed, small) ===\n" + "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        return {
            "facts": len(self.facts),
            "addresses": len(self.by_address),
            "unbound_now": len(self.unbound()),
            "n_action_write": self.n_action_write,
            "n_unbound_minted": self.n_unbound_minted,
            "n_query": self.n_query,
            "n_query_hit": self.n_query_hit,
            "n_query_miss": self.n_query_miss,
            "n_frames_served_off_context": self.n_frames_served_off_context,
            "n_serve_empty_not_yet_observed": self.n_serve_empty_not_yet_observed,
            "n_spec_minted": self.n_spec_minted,
            "n_stage_active": self.n_stage_active,
            "n_stage_verified": self.n_stage_verified,
            "n_spec_unopened": sum(
                1 for f in self.facts if f.source == "spec" and f.mint_step == _NOT_YET
            ),
            "lookahead": LOOKAHEAD,
            "serve_max": SERVE_MAX,
        }

    def render_addresses(self) -> str:
        """Just the address list, for the `list_unbound` tool."""
        unb = self.unbound()
        if not unb:
            return "list_unbound: every observed address has been resolved."
        return "unresolved addresses:\n" + "\n".join(
            f"- {f.address}   [step {f.mint_step}, "
            f"{len(self.available_frames(f, set()))} frame(s) available]"
            for f in unb
        )

    def render_all(self, cap: int = 12) -> str:
        """The whole bank, capped, for the `list_facts` tool."""
        if not self.facts:
            return "list_facts: the bank is empty."
        out = []
        for f in self.facts[-cap:]:
            mark = "RESOLVED" if f.resolved else ("UNRESOLVED" if not f.value else "KNOWN")
            out.append(f"- [{mark}] {f.address} = {f.value!r} @step {f.mint_step}")
        head = f"facts (last {min(cap, len(self.facts))} of {len(self.facts)}):"
        return head + "\n" + "\n".join(out)
