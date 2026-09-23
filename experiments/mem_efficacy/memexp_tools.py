"""mem_efficacy / the PULL arm's tool module.

WHY TOOLS ARE REGISTERED RATHER THAN DISPATCHED
-----------------------------------------------
The planning layer in `memexp_bind` knows nothing about memory. It knows how to offer a set of
tool specs to a model, run whichever one comes back, and append the result to the same
conversation. Everything domain-specific lives here. Adding a tool is therefore a change to
this file alone, and the loop cannot develop a special case for any particular one.

WHY A TOOL RESULT CARRIES BOTH TEXT AND FRAMES
----------------------------------------------
This is the fix for the defect that made PMH's read path a two-hop sequence. There,
`search_memory` docstring says it plainly -- "Returns TEXT ONLY ... Never loads images" -- so a
Planner that needed to SEE something had to call `search_memory`, read a segment id out of the
text, and then call `retrieve_visual(segment_id=...)`. Two model decisions per fact, in a regime
where the model picks `none` most of the time, so the probability of completing the sequence is
the product of two small numbers.

A tool here returns the claim, its provenance, AND the frames in one call. One decision, one
round trip.

WHAT THE TOOLS DELIBERATELY DO NOT DO
-------------------------------------
`query_world` does not answer the question. It returns the observation and lets the Planner --
which is a VLM -- look at it. Writing a text answer would require either a vision model call
(expensive, and a second place for hallucination to enter the belief state) or the harness
guessing from the primitive text (which cannot know what was inside a drawer). So the substrate
stores where to look, and the tool carries the picture.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from memexp_substrate import Substrate

# ---------------------------------------------------------------------------------------
# THE OUTPUT CONTRACT. Read this before editing either string below.
#
# The Planner's own planning instruction ends with, verbatim:
#     "Output strict JSON with exactly two fields: current_primitive and keyframe_positions."
# and `harness/vlm_output_parser.parse_vlm_output` reads EXACTLY those two key names:
#     parsed.get("current_primitive", parsed.get("current_subtask", ""))
#
# This module previously taught the model to finish with `{"primitive": "..."}`. That key is
# parsed to the EMPTY STRING -- `json.loads` succeeds, so no exception is raised and the
# prose-fallback branch is never taken: the plan step simply yields no primitive. Measured with
# the real parser: `{"primitive": "open middle drawer"}` -> `primitive=''`.
#
# A tool call is genuinely a third form, and the model must be TOLD it is the sanctioned
# exception. That is not "leaning on the model": without it, the arm asks the model to follow a
# rule and break it in the same prompt, and a model that obeys instructions declines -- which is
# what a 2.84% call rate (5 calls / 176 plan steps, job 592860) actually measured.
#
# `validate_arm.py` CHECK 15 re-reads the planner's source and asserts these strings agree with
# it, so the two cannot drift apart again in silence.
# ---------------------------------------------------------------------------------------
FINISH_FORM = '{"current_primitive": "<your chosen primitive>", "keyframe_positions": [<positions>]}'

FORCE_FINAL = (
    "TOOLS ARE NOW CLOSED for this planning step. Do not output a tool call. "
    "Output exactly one JSON object with exactly two fields: "
    '{"current_primitive": "<your chosen primitive>", "keyframe_positions": [<positions>]}.'
)


@dataclass
class ToolResult:
    """What a tool call produces.

    `text` is what the model reads; `frames` are absolute frame indices to attach as images;
    `meta` is for the census and never reaches the model.
    """
    text: str
    frames: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class Tool:
    def __init__(
        self,
        name: str,
        args: dict[str, str],
        description: str,
        fn: Callable[[dict[str, Any]], ToolResult],
    ) -> None:
        self.name = name
        self.args = args          # arg name -> short description
        self.description = description
        self.fn = fn

    def spec_line(self) -> str:
        arg_txt = ", ".join(f'"{k}": "<{v}>"' for k, v in self.args.items())
        return f'  {{"tool": "{self.name}", {arg_txt}}}\n      {self.description}'


# The model's output is parsed leniently and NEVER treated as prose. A tool call is recognised
# only when the JSON names a registered tool; anything else is passed through unchanged to the
# caller, whose own parser decides what it was. That ordering matters: if this module claimed
# ambiguous outputs, it would silently swallow primitives and the arm would score zero for a
# reason that looks exactly like "the Planner produced nothing".
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_call(text: str, known: set[str]) -> dict[str, Any] | None:
    """Return the tool call in `text`, or None if this is not one (i.e. it is a primitive).

    Tolerates a markdown fence and a reasoning block, for the same reason
    `harness.vlm_output_parser` does: a writer that wraps its JSON must not be read differently
    by two components.
    """
    s = str(text or "").strip()
    if "</think>" in s:
        s = s[s.rfind("</think>") + len("</think>"):].strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    m = _JSON_OBJ.search(s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("tool", "")).strip().lower()
    if name not in known:
        return None
    args = obj.get("args")
    if not isinstance(args, dict):
        # Also accept flattened arguments, because a model told `{"tool":..., "args":{...}}` will
        # sometimes emit the keys at the top level instead. Rejecting that costs a wasted round
        # trip and teaches the model nothing.
        args = {
            k: v for k, v in obj.items()
            if k not in {"tool", "args", "reason", "thought", "why"}
        }
    return {"tool": name, "args": args}


class MemoryTools:
    """The registry. One per episode, holding the substrate it reads from."""

    # Opts this registry into `memexp_bind.verify_binding`'s contract-consistency assertions.
    # Must be a CLASS attribute: the binding interrogates the instance, and a module-level
    # declaration alone reads as "off" there -- a gate switched off while looking switched on.
    CONTRACT_AWARE = True

    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate
        self.calls: list[dict[str, Any]] = []
        self._served: dict[str, ToolResult] = {}
        # The DENOMINATOR for the call rate. Without it, "the Planner called 5 tools" has no
        # scale: 5 out of 176 plan steps and 5 out of 5 are the same sentence. An arm that never
        # showed the block at all would otherwise be reported as "the model declined".
        self.n_spec_shown = 0
        self.tools: dict[str, Tool] = {
            t.name: t for t in (
                Tool(
                    "query_world",
                    {"address": "an address from the pushed memory, e.g. contents(middle drawer)"},
                    "Open the observation behind an address. Returns what was recorded AND the "
                    "frames to look at. Use when a fact cannot be determined from what you "
                    "already see.",
                    self._query_world,
                ),
                Tool(
                    "list_unbound",
                    {},
                    "List every address that was observed but not yet resolved.",
                    self._list_unbound,
                ),
                Tool(
                    "list_facts",
                    {},
                    "List what the memory already records, with resolution state.",
                    self._list_facts,
                ),
            )
        }

    # ---- spec ---------------------------------------------------------------------------
    def spec_text(self) -> str:
        """The tool block appended to the prompt. Wording is deliberately NEUTRAL.

        No default, no "prefer", no "avoid habitual". PMH's three prompt variants each ended with
        `Default: tool none` / `avoid habitual search_memory` / `prefer none`, and the measured
        outcome was 0 `search_memory` calls and 85% `none` decisions -- a read path that was
        described as agent-controlled and had no agent in it (`pmh_memory.py:2850-2854`). A
        prompt that leans on the model does not measure the model.

        Clarity about the OUTPUT CONTRACT is a separate matter from pressure: see `FINISH_FORM`
        above. Telling the model that a tool call is exempt from the two-field rule removes a
        contradiction; it does not tell the model which way to answer.
        """
        lines = [
            "=== memory tools (optional, available at any point) ===",
            "A TOOL CALL IS THE ONE ALLOWED EXCEPTION to this step's two-field output rule. When "
            "you need something you cannot see, output a tool call INSTEAD of the two-field "
            "object; the evidence comes back immediately and you then finish the step in the "
            "two-field form. Emit the primitive when you are ready; there is no penalty for "
            "calling a tool and no reward for avoiding one.",
            "To call one, output STRICT JSON and nothing else:",
        ]
        for t in self.tools.values():
            lines.append(t.spec_line())
        lines.append(
            "  " + FINISH_FORM + "    <- this ends the planning step"
        )
        lines.append(
            "The finishing form uses the SAME two fields your planning instruction names. Do not "
            "invent other key names: `primitive` and `current_primitive` are not the same field, "
            "and an object with the wrong key name is not a valid answer."
        )
        self.n_spec_shown += 1
        return "\n".join(lines)

    # ---- dispatch -----------------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                text=f"unknown tool {name!r}; available: {sorted(self.tools)}",
                meta={"ok": False, "reason": "unknown_tool"},
            )
        # Cache by the SAME normalised arguments within one plan step. A Planner that asks the
        # same question twice has not received a useful answer, and serving it again spends a
        # round trip to teach it nothing. Counted rather than hidden: a non-zero repeat rate is a
        # signal that the first result was unreadable, which is a defect in this module, not in
        # the model.
        key = name + "|" + json.dumps(args, sort_keys=True, default=str)
        if key in self._served:
            prev = self._served[key]
            self.calls.append({"tool": name, "args": args, "repeat": True})
            return ToolResult(
                text="(already retrieved this planning step; same result follows)\n" + prev.text,
                frames=[],   # frames are already in the conversation; re-attaching them would
                             # make the same pixels look like new evidence
                meta={"ok": True, "repeat": True, "frames": 0},
            )
        result = tool.fn(args)
        self._served[key] = result
        self.calls.append({
            "tool": name, "args": args, "repeat": False,
            "ok": bool(result.meta.get("ok", True)),
            "frames": len(result.frames),
        })
        return result

    def reset_step_cache(self) -> None:
        self._served.clear()

    # ---- the tools themselves -----------------------------------------------------------
    def _query_world(self, args: dict[str, Any]) -> ToolResult:
        address = str(args.get("address", args.get("query", args.get("what", "")))).strip()
        if not address:
            return ToolResult(
                text="query_world needs an address, e.g. contents(middle drawer). "
                     "Call list_unbound to see the addresses that exist.",
                meta={"ok": False, "reason": "no_address"},
            )
        facts, how = self.substrate.query(address)
        if not facts:
            # A miss is reported as a miss, and it names what DOES exist. A tool that returns a
            # plausible-looking empty success is how a read path ends up "looking healthy while
            # doing nothing", which this project has already paid for three times.
            return ToolResult(
                text=f"query_world({address!r}): no address matched.\n"
                     + self.substrate.render_addresses(),
                meta={"ok": False, "reason": "miss", "how": how, "frames": 0},
            )
        lines = [f"query_world({address!r}) -> {how}; {len(facts)} record(s):"]
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
            # Two different reasons, reported differently -- and getting this branch right matters
            # more than it looks. "Already on context" is the Planner's own doing and needs no
            # action. "Not observed yet" is a timing fact, and a Planner that mistakes it for
            # "empty" will place the object in the wrong container. Saying the wrong one is
            # therefore not a cosmetic bug; it manufactures a hallucination.
            if any(not self.substrate.window_elapsed(f) for f in facts):
                lines.append(
                    "No frames attached: the observation this address refers to has not been "
                    "made yet (the frames are still ahead of the current step). Do not conclude "
                    "the container is empty from this. It becomes readable a few steps later; "
                    "ask again then, or act on what you can see now."
                )
            else:
                lines.append(
                    "No new frames attached: every frame for this address is already in your "
                    "context. Re-read the images you already have."
                )
        return ToolResult(
            text="\n".join(lines),
            frames=frames,
            meta={"ok": True, "how": how, "records": len(facts), "frames": len(frames)},
        )

    def _list_unbound(self, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(text=self.substrate.render_addresses(), meta={"ok": True, "frames": 0})

    def _list_facts(self, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(text=self.substrate.render_all(), meta={"ok": True, "frames": 0})

    def stats(self) -> dict[str, Any]:
        by_tool: dict[str, int] = {}
        repeats = 0
        off_context_frames = 0
        for c in self.calls:
            by_tool[str(c["tool"])] = by_tool.get(str(c["tool"]), 0) + 1
            if c.get("repeat"):
                repeats += 1
            off_context_frames += int(c.get("frames") or 0)
        return {
            "n_calls": len(self.calls),
            "by_tool": by_tool,
            "n_repeat_calls": repeats,
            "n_frames_served": off_context_frames,
            "n_spec_shown": self.n_spec_shown,
        }
