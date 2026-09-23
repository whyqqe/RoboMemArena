"""mem_efficacy / the SEARCH variant of the tool registry.

WHY THIS FILE EXISTS, IN ONE SENTENCE
-------------------------------------
The previous pull arm measured a 2.84% call rate (5 calls / 176 plan steps, job 592860) and the
cause is not that the model is unwilling -- it is that the arm asked for something the model had
been told not to do, and asked in a way it could not answer.

THE ROOT CAUSE, AND WHY IT IS NOT A PROMPT-TUNING PROBLEM
---------------------------------------------------------
The Planner is given a hard output contract, and it is the LAST instruction in the user turn
(`api_vlm_planner.py`: *"Output strict JSON with exactly two fields: current_primitive and
keyframe_positions."*). A tool call is not two fields. So the arm was, in effect, instructing the
model to follow a rule and then asking it to break that rule -- and a model that obeys instructions
declines. `memexp_tools.spec_text()` even said "you may call a tool instead of emitting a
primitive", but it never said that this is the ONE case where the two-field rule does not apply, and
it then made the conflict worse by teaching the finishing form as `{"primitive": "..."}` -- a THIRD
key name, used again in `memexp_bind._FORCE_FINAL`.

So three mutually incompatible output contracts arrived in one prompt:

    {"current_primitive": ..., "keyframe_positions": [...]}   <- the planning instruction
    {"tool": "<name>", "args": {...}}                         <- the tool block
    {"primitive": "<...>"}                                    <- the tool block's terminator
                                                                AND the forced-finish message

This file repairs the contract rather than arguing with the model about it: the tool call is
declared to be the exception, and the finishing form uses the SAME two key names the planner's own
instruction names. `selftest_search.py` asserts that agreement by re-reading the planner's prompt
text, so the two cannot drift apart silently again.

THE SECOND CAUSE: THE TOOL COULD NOT BE USED WITHOUT ALREADY KNOWING THE ANSWER
-------------------------------------------------------------------------------
`query_world` takes an ADDRESS -- an exact string like `contents(middle drawer)` that the model must
copy out of the pushed memory block. That inverts the purpose of retrieval. A model that can name
the address usually can already see what it refers to; a model that cannot name it has no way to
reach the record at all. The measured evidence for this is in the same artifact: 5 calls, all of
them on addresses that were already advertised.

`search_memory(query)` removes the precondition. The query is FREE TEXT in the model's own words
("which drawer did I open last", "pour tomato sauce"), scored against every record's address and
value, and it returns the matching records plus their frames. `query_world` is kept, because when
the model DOES know the address it is the more precise instrument.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not make the decision for the model. There is no default, no forced call, and no
"prefer". The gate in `memexp_bind` is unchanged: the model chooses, and the call rate is counted.
The whole point of fixing the wording is to make that choice MEASURABLE -- a rate of 2.84% under a
self-contradicting prompt measures the prompt, not the model.
"""
from __future__ import annotations

import os
import re
from typing import Any

from memexp_tools import MemoryTools as _BaseTools
from memexp_tools import Tool, ToolResult
# Re-exported so this module is a DROP-IN registry. `memexp_bind.load_registry` validates that a
# registry exports both `MemoryTools` and `parse_tool_call`, and the tool loop parses the model's
# output through the SAME parser the registry ships -- a variant that omitted it would fail to
# load rather than quietly fall back to a different reader. (It did exactly that on first run:
# the loader refused `memexp_tools_search` with "does not export ['parse_tool_call']", which is the
# check earning its place.)
from memexp_tools import parse_tool_call  # noqa: F401 - public re-export, used by the loop

# Matches the planner's own instruction. If `api_vlm_planner.py` ever changes its contract, the
# selftest that compares these two strings fails rather than letting the arms diverge in silence.
FINISH_FORM = '{"current_primitive": "<your chosen primitive>", "keyframe_positions": [<positions>]}'

# Opts this registry into the contract-consistency assertions in `memexp_bind.verify_install`.
# The ORIGINAL registry does not set this, and must not: it has the defect those assertions detect,
# so asserting them there would fail a completed arm's preflight instead of repairing anything. The
# flag is what lets the fix be verified without rewriting the arm it was diagnosed from.
#
# It is a MODULE constant for the record and a CLASS attribute because that is where the binding
# reads it from. Declaring it only on the module would leave the gate switched off while looking
# switched on -- the exact "configured but inert" failure this project keeps paying for.
CONTRACT_AWARE = True

FORCE_FINAL = (
    "TOOLS ARE NOW CLOSED for this planning step. Do not output a tool call. "
    "Output exactly one JSON object with exactly two fields: "
    '{"current_primitive": "<your chosen primitive>", "keyframe_positions": [<positions>]}.'
)

# Bounded: a search that can return the whole bank is a push in a tool's clothing, and it would
# also make the prompt grow without limit -- the failure mode a previous experiment mis-read as
# "the correction rendered nothing".
SEARCH_MAX = int(os.environ.get("MEMEXP_SEARCH_MAX", "4"))


def _max_rounds() -> int:
    """The loop's own fuse, read from the same variable the fuse reads.

    `memexp_bind._max_rounds()` enforces this; the spec has to state it. Two readers of one
    variable is the point: the previous spec promised an unbounded budget next to a fuse of 3, and
    the Planner believed the spec.
    """
    return max(1, int(os.environ.get("MEMEXP_PULL_MAX_ROUNDS", "3")))

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "for", "with", "from", "into",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "do", "did", "does", "has", "have", "had", "what", "which", "when", "where", "how", "why",
    "all", "any", "some", "then", "than", "there", "here", "you", "your", "i", "me", "my",
    "there", "inside", "contain", "contains", "contents", "empty", "already", "last",
}


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with underscores and parens split.

    Splitting on `(`/`)` is what makes the address form indexable: `contents(middle drawer)` has to
    be findable by "drawer" and by "middle", otherwise a model that asks in ordinary words can never
    reach a record and the tool is back to requiring the exact string.
    """
    parts = re.split(r"[^0-9a-z]+", str(text or "").lower().replace("_", " "))
    return [p for p in parts if len(p) >= 3 and p not in _STOP]


class MemoryTools(_BaseTools):
    """The search variant. Same substrate, same dispatch, one more instrument and an honest prompt.

    Subclassed rather than copied so the two variants cannot drift: `query_world`, `list_unbound`,
    `list_facts` and the miss-reporting rules are the previous arm's, verbatim, including the
    "not observed yet" branch that took two attempts to get right. Only the tool set and the spec
    differ, which is exactly the variable this arm is meant to isolate.
    """

    # Read from the INSTANCE, because that is what the binding interrogates. See the module-level
    # note above: a module-only declaration reads as "off" here and would silence the gate.
    CONTRACT_AWARE = True

    def __init__(self, substrate) -> None:
        super().__init__(substrate)
        self.tools["search_memory"] = Tool(
            "search_memory",
            {"query": "what you want to find, in your own words"},
            "Search EVERYTHING the episode has established -- containers opened, objects touched, "
            "stages completed -- and return the matching records with the frames that show them. "
            "Use this when you do NOT know the exact address: describe what you want in ordinary "
            "words and the search finds the record.",
            self._search_memory,
        )
        # Instrumentation for the question the arm exists to answer: was the tool OFFERED, was it
        # CALLED, and did the search reach anything. `n_spec_shown` comes from the base class --
        # it is the call rate's denominator and every tool registry needs it.
        self.n_search = 0
        self.n_search_hit = 0
        self.n_search_miss = 0

    # ---- spec ---------------------------------------------------------------------------
    def spec_text(self) -> str:
        text = "\n".join([
            "=== memory tools ===",
            'Your planning instruction says "exactly two fields". A TOOL CALL IS THE ONE ALLOWED '
            "EXCEPTION to that: when you need something you cannot see, output a tool call "
            "INSTEAD of the two-field object. The "
            "evidence comes back immediately and you then finish the step in the two-field form. "
            # THE BUDGET IS STATED, BECAUSE THE PREVIOUS TEXT DID NOT STATE IT AND THE CODE
            # ENFORCED A DIFFERENT ONE. The spec said "as many times as you need", `_max_rounds()`
            # capped the loop at 3, and job 595132 measured `cap_hits=2`: the Planner was told it
            # could call freely, then cut off. A spec the implementation contradicts makes
            # `cap_hits` a measurement of the spec bug rather than of the model's behaviour, so the
            # number is read from the same variable the fuse reads.
            f"You may call up to {_max_rounds()} tool(s) within one planning step; a further call "
            "is refused and you will be told to finish the step. There is no penalty for calling a "
            "tool and no reward for avoiding one.",
            "To call one, output STRICT JSON and nothing else:",
            '  {"tool": "search_memory", "args": {"query": "<what you want to find, in your own '
            'words>"}}\n'
            "      Searches every record the episode has established: which containers were opened, "
            "what was touched, which stages are done. THIS IS THE ONE TO USE WHEN YOU DO NOT KNOW "
            "THE EXACT ADDRESS.",
            '  {"tool": "query_world", "args": {"address": "<an address you have already seen>"}}\n'
            "      Opens the observation behind an address shown in the memory block above. More "
            "precise than search_memory, but it needs the exact address -- so use it only when you "
            "can see that address.",
            # `list_unbound` AND `list_facts` ARE NOT OFFERED, AND THE REASON IS MEASURED. They are
            # pure enumerations -- they return the address inventory and the resolution states, and
            # never an observation. The push block rendered into this prompt on EVERY step already
            # carries that inventory (`render_push`: recent actions, then the ready / deferred /
            # upcoming addresses), so a call to either one spends a full model round-trip asking for
            # something the model was handed. Job 595132 measured the cost: 41 of 95 tool calls
            # (43%) were `list_facts` and 6 more were `list_unbound`, together 49 calls that
            # returned zero observation frames out of a total of 19 served. Removing them from the
            # spec is not removing the capability -- both stay registered and still answer if called
            # -- it stops the prompt from soliciting work whose answer it has already given.
            "",
            "CALL A TOOL WHENEVER ANY OF THESE HOLDS:",
            "  * the next action depends on what is inside a container that is not currently "
            "visible;",
            "  * the instruction needs a count -- how many pours, how many objects placed -- and "
            "the whole episode is not visible;",
            "  * you cannot tell which stages are already finished;",
            "  * the same primitive has repeated for many steps and what you can see does not "
            "explain why.",
            "If none of those hold and the current frames are enough, do NOT call a tool.",
            "",
            "Example A -- searching, then finishing:",
            '  you:    {"tool": "search_memory", "args": {"query": "<what you want to find>"}}',
            "  result: 2 record(s): - address=contents(<container>) value=<NEEDS LOOKING: value not "
            "recorded> observed_at_step=<n> source=action",
            "          Attached: 2 frame(s) NOT currently in your context (indices [<a>, <b>]). "
            "These are the observations themselves -- read them.",
            "  you:    " + FINISH_FORM,
            "Example B -- finishing directly, with no tool call:",
            '  you:    {"current_primitive": "<the primitive you choose>", "keyframe_positions": []}',
            "",
            "NOTE the finishing form: the SAME two fields your planning instruction names. Do not "
            "invent other key names -- `primitive` and `current_primitive` are not the same field, "
            "and an object with the wrong key name is not a valid answer.",
        ])
        # Incremented here as well as in the base class, because this override never calls
        # `super().spec_text()`. Missing it would leave the call rate's denominator at zero for
        # exactly the arm whose call rate is the point of the experiment.
        self.n_spec_shown += 1
        return text
    # ---- the tool ------------------------------------------------------------------------
    def _search_memory(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", args.get("q", args.get("what", "")))).strip()
        if not query:
            return ToolResult(
                text="search_memory needs a query, e.g. "
                     '{"tool": "search_memory", "args": {"query": "drawer contents"}}. '
                     "Pass your own words; the exact address is not required.",
                meta={"ok": False, "reason": "no_query"},
            )
        self.n_search += 1
        facts, how = self._rank(query)
        if not facts:
            # A MISS STILL HAS TO BE AN ANSWER, or the tool is not a retrieval tool. `_rank`
            # requires literal token overlap (`if not overlap: continue`) and reported a bare miss
            # otherwise -- so a Planner asking in its own words got "nothing matched" whenever its
            # phrasing did not literally appear in an address or value. Measured on job 595132: 12
            # of 25 searches (48%) missed, and the reply to each was `render_addresses()`, i.e. an
            # enumeration of what the model could already see. Half of the arm's highest-value path
            # returned no observation, and the other half of that exchange was spent on a listing.
            #
            # The Planner's real question -- "what do I remember about this?" -- is answerable
            # whenever the episode holds a readable record, so the fallback answers it with the most
            # recent ones. `how` is echoed into the reply, so `recency_fallback` is distinguishable
            # from a ranked `search` by the model and by the census, without either trusting prose.
            facts, how = self._recent_readable()
            if not facts:
                self.n_search_miss += 1
                return ToolResult(
                    text=f"search_memory({query!r}): this episode remembers nothing readable yet -- "
                         f"no record has an observation that can be shown. That is a statement about "
                         f"the episode, not about the world; ask again after the actions you have "
                         f"planned have been carried out.\n" + self.substrate.render_addresses(),
                    meta={"ok": False, "reason": "no_records", "how": how, "frames": 0},
                )
        self.n_search_hit += 1
        lines = [f"search_memory({query!r}) -> {how}; {len(facts)} record(s), best match first:"]
        for f in facts:
            lines.append(
                f"- address={f.address} value={f.value or '<NEEDS LOOKING: value not recorded>'}"
                f" observed_at_step={f.mint_step} source={f.source}"
            )
        frames = self.substrate.serve(facts, on_context=args.get("_on_context"))
        if frames:
            lines.append(
                f"Attached: {len(frames)} frame(s) NOT currently in your context "
                f"(indices {frames}). These are the observations themselves -- read them."
            )
        else:
            # The same two-way branch `query_world` uses, for the same reason: "already in your
            # context" is the model's own doing, while "not observed yet" is a timing fact that a
            # model misreading as "empty" will turn into a wrong placement.
            if any(not self.substrate.window_elapsed(f) for f in facts):
                lines.append(
                    "No frames attached: the observation these records refer to has not been made "
                    "yet (the frames are still ahead of the current step). Do NOT conclude the "
                    "container is empty from this. It becomes readable a few steps later; ask "
                    "again then, or act on what you can see now."
                )
            else:
                lines.append(
                    "No new frames attached: the frames for these records are already in your "
                    "context. Read the records above; do not re-ask for the same thing."
                )
        return ToolResult(text="\n".join(lines), frames=frames,
                          meta={"ok": True, "how": how, "n_facts": len(facts),
                                "frames": len(frames)})

    def _rank(self, query: str, k: int = SEARCH_MAX) -> tuple[list, str]:
        """Token-overlap ranking over every record's address AND value. Deterministic.

        No embeddings and no sampling anywhere: two identical runs must retrieve identical
        evidence, or the arm's score carries variance that has nothing to do with the memory
        mechanism. Ties break on recency, then on insertion order.

        `value` is indexed as well as `address` because the two carry different information here:
        an action mints `contents(middle drawer)` with an EMPTY value (the answer needs looking at),
        while a stage record mints `address="stage"` with the stage NAME as its value. Indexing only
        addresses would make every stage unfindable.
        """
        qt = set(tokens(query))
        if not qt:
            return [], "empty_query"
        corpus = [f"{f.address} {f.value}" for f in self.substrate.facts]
        doc_tok = [set(tokens(c)) for c in corpus]
        n_docs = max(1, len(doc_tok))
        df: dict[str, int] = {}
        for toks in doc_tok:
            for t in toks:
                df[t] = df.get(t, 0) + 1
        scored: list[tuple[float, int, Any]] = []
        for i, (fact, toks) in enumerate(zip(self.substrate.facts, doc_tok)):
            overlap = qt & toks
            if not overlap:
                continue
            # Smoothed inverse document frequency, positive and without log(0). A word that appears
            # in every record ("stage") must not outweigh the one that discriminates.
            score = sum(1.0 + (n_docs / max(1, df[t])) for t in overlap)
            scored.append((score, i, fact))
        if not scored:
            return [], "miss"
        scored.sort(key=lambda x: (-x[0], -x[2].mint_step, x[1]))
        best = scored[0][0]
        # Keep everything tied with the top score, then cap. Returning only the single best row
        # would hide the sibling containers, which is precisely what a "which drawer" question
        # needs -- all three drawer records are the same answer until the frames are read.
        picked = [f for s, _, f in scored if s >= best * 0.999][:k]
        return picked, "search"

    def _recent_readable(self, k: int = SEARCH_MAX) -> tuple[list, str]:
        """The most recent records that can actually show an observation right now.

        The fallback arm of `search_memory`, used when no record shares a token with the query.
        Ranked by recency rather than by a score, because with zero overlap there is nothing to
        score; and FILTERED on `available_frames`, because a record whose observation window has
        not closed returns no pixels -- selecting on recency alone would hand back an empty frame
        list and reproduce exactly the dead end this fallback exists to remove.
        """
        cands = [f for f in self.substrate.facts if self.substrate.available_frames(f, set())]
        cands.sort(key=lambda f: -int(getattr(f, "mint_step", 0) or 0))
        return cands[:k], "recency_fallback"

    def stats(self) -> dict[str, Any]:
        out = dict(super().stats())
        out.update({
            "n_spec_shown": self.n_spec_shown,
            "n_search": self.n_search,
            "n_search_hit": self.n_search_hit,
            "n_search_miss": self.n_search_miss,
            "search_max": SEARCH_MAX,
        })
        return out
