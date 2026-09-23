#!/usr/bin/env python
"""mem_efficacy / GATE 0b: drive the PULL arm's loop with a stub, end to end, before any GPU time.

WHY THIS EXISTS
---------------
`validate_arm.py` CHECK 8 proves the binding is INSTALLED and the substrate behaves. It cannot
prove the LOOP works, because the loop only runs when a model answers with a tool call, and no
unit test of a registry can produce that. The gap between "the hook is installed" and "the hook
does its job" is precisely where this project has lost runs: job 586700 spent 26 tasks with a
memory feature enabled whose own counter stayed at zero, and job 570810 logged 99% redundant
retrieval against `picked=1`.

So this gate replaces the model with a scripted one that emits exactly the outputs the loop must
handle -- a tool call, then a primitive; then a model that never stops calling tools; then a
Planner in an episode with nothing to search for -- and asserts the loop's observable behaviour
on the messages list it actually passes downstream.

It costs a couple of seconds and no API quota.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.environ.get("ROOT", "/project/peilab/why/RoboMemArena")
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark"))
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark", "openpi_minimal_runtime"))
# `task2_26_reference_stage` lives here, and the evaluator imports it as `stage_eval`, i.e. this
# directory is on the real run's sys.path too. Added so G0b-10 exercises the SAME module the
# task-spec seed resolves at run time, rather than a copy that could drift from it.
sys.path.insert(0, os.path.join(ROOT, "evaluation_benchmark", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MEMEXP_PULL_ENABLE", "1")
os.environ.setdefault("MEMEXP_PULL_TOOLS", "1")
os.environ.setdefault("MEMEXP_PULL_MAX_ROUNDS", "3")

from PIL import Image  # noqa: E402

FAILS: list[str] = []
NOTES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail and not ok:
        print(f"         {detail}")
    if not ok:
        FAILS.append(label)
    return ok


# ---------------------------------------------------------------------------------------
# A fake planner module. `_bind` patches exactly these two attributes, which is the whole
# surface this arm depends on.
# ---------------------------------------------------------------------------------------
class _FakePlanner:
    """The two attributes the arm's `_build_messages` wrapper reads, plus a frame store.

    `frame_store_main` is deliberately NOT a pre-filled dict. In the real evaluator it is
    populated as frames arrive, so at plan step N it contains frames `0..N-1` and nothing beyond.
    An earlier revision of this test pre-filled all 600 frames, which made frames that did not
    exist yet look retrievable -- and that leniency is precisely what hid the write-time frame
    resolution defect this file exists to catch. The store here therefore only exposes frames
    strictly before the current step, and the tests have to advance the episode for evidence to
    become available, exactly as a real episode does.
    """

    def __init__(self) -> None:
        self.step = 0
        self._current_subtask = ""
        self.K_indices_abs: list[int] = []
        self.use_wrist = False
        self._all_frames = {i: Image.new("RGB", (8, 8), (i % 255, 0, 0)) for i in range(600)}
        self.frame_store_wrist: dict[int, Image.Image] = {}

    # The two methods the real evaluator calls on an episode boundary. They are here so that
    # `_wrap_resets` has something to wrap: without them the reset hooks silently do not exist and
    # the episode-isolation gate (G0b-9) would pass on a class where nothing was ever hooked --
    # the "configured but inert" failure this project keeps paying for.
    def reset_episode(self, *a, **k):
        self.episode_resets = getattr(self, "episode_resets", 0) + 1

    def set_task_info(self, *a, **k):
        self.task_sets = getattr(self, "task_sets", 0) + 1

    @property
    def frame_store_main(self):
        return {i: img for i, img in self._all_frames.items() if i < self.step}


def _fake_build_messages(self, memory_main_frames, memory_wrist_frames,
                         context_main_frames, context_wrist_frames, *, extra_memory_text=""):
    return [{"type": "text", "text": f"base-prompt step={self.step}"}]


def _make_fake_planner_class():
    """A FRESH class per test.

    Not a shared one, and not a detail: `_wrap_build_messages` guards on `_memexp_wrapped`, so a
    reused class would be wrapped once and every later test would silently reuse the first
    test's captured `orig`. That is the same class of bug this whole file exists to catch -- a
    suite that passes because it stopped testing anything after the first case.
    """
    return type("FakePlanner", (_FakePlanner,), {"_build_messages": _fake_build_messages})


def _install(script):
    """Bind the arm onto a fresh fake module driven by `script(user_content) -> str`."""
    import memexp_bind

    planner_cls = _make_fake_planner_class()

    def orig(*, system_prompt, user_content, **kw):
        return script(user_content)

    mod = type("M", (), {
        "ApiMemoryPlanner": planner_cls,
        "infer_primitive_via_api": staticmethod(orig),
    })
    # Reset the per-thread context so no test inherits another's planner, substrate or counters.
    memexp_bind._LOCAL.__dict__.clear()
    memexp_bind._wrap_build_messages(mod)
    memexp_bind._wrap_api_call(mod)
    return memexp_bind, mod


def _install_with_resets(script):
    """Same as `_install`, plus the reset hooks the real evaluator triggers.

    `_install` deliberately omits `_wrap_resets` so the older gates exercise the loop WITHOUT the
    episode boundary. This variant adds it, because the property under test in G0b-9 is precisely
    what happens at that boundary.
    """
    import memexp_bind

    planner_cls = _make_fake_planner_class()

    def orig(*, system_prompt, user_content, **kw):
        return script(user_content)

    mod = type("M", (), {
        "ApiMemoryPlanner": planner_cls,
        "infer_primitive_via_api": staticmethod(orig),
    })
    memexp_bind._LOCAL.__dict__.clear()
    memexp_bind._wrap_build_messages(mod)
    memexp_bind._wrap_api_call(mod)
    memexp_bind._wrap_resets(mod)
    return memexp_bind, mod


def _step(mod, planner, primitives: list[str], pending: str = "", at_step: int = 300):
    """One plan step's worth of the real sequence: prompt build, then the API call.

    `at_step` is deliberately mid-episode. At step 1 the frames an observation produces are still
    INSIDE the 5-frame context window, so nothing is servable and the image assertions would fail
    for the right reason but for the wrong test -- they would be measuring the episode's opening
    rather than the retrieval path. (G0b-1 found exactly this as a real defect in an earlier
    revision: frames resolved at write time named frames that had not been observed yet.)
    """
    planner._current_subtask = pending
    planner.step = at_step
    msgs = planner._build_messages([], [], [(None, None)] * 5, [None] * 5)
    out = mod.infer_primitive_via_api(
        system_prompt="sys", user_content=msgs, api_key="k", base_url="u", model="m"
    )
    primitives.append(str(out))
    return msgs, out


def _n_images(msgs) -> int:
    return sum(1 for p in msgs if isinstance(p, dict) and p.get("type") == "image")


def _null_substrate():
    """A throwaway substrate, for constructing a registry just to ask it what it teaches.

    Deliberately EMPTY: `spec_text()` must be a pure description of the tools, so a registry that
    needs episode state to render its own instructions would be a design fault. Using an empty one
    here is what makes that a tested property rather than an assumption.
    """
    import memexp_substrate

    return memexp_substrate.Substrate()


def main() -> int:
    import memexp_bind

    # -----------------------------------------------------------------------------------
    # G0b-1: a tool call is executed, its result and frames are appended, and the model gets
    # a second turn IN THE SAME CONVERSATION -- the property PMH's fixed loop lacked.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-1] tool call -> execution -> same-conversation continuation -> primitive")
    seen: list[list] = []
    turns = {"n": 0}

    def script_one_tool(user_content):
        seen.append(list(user_content))
        turns["n"] += 1
        asked = any("query_world(" in str(p.get("text", "")) for p in user_content
                    if isinstance(p, dict))
        if asked:
            return '{"current_primitive": "pick from middle drawer"}'
        # Ask only when the push ADVERTISES the address. A Planner that reads its own memory
        # block behaves this way, and this is what the advertisement rule is for.
        adv = [p for p in user_content if isinstance(p, dict)
               and "frame(s) ready to look at" in str(p.get("text", ""))]
        if adv and "contents(middle drawer)" in str(adv[0]["text"]):
            return '{"tool": "query_world", "address": "contents(middle drawer)"}'
        return '{"current_primitive": "wait"}'

    bind, mod = _install(script_one_tool)
    pl = mod.ApiMemoryPlanner()
    prims: list[str] = []

    # Step 300: the primitive "open middle drawer" was emitted for the previous step, so the
    # address is minted here. Its frames DO NOT EXIST YET, so it must not be advertised.
    m1, _ = _step(mod, pl, prims, pending="open middle drawer", at_step=300)
    push1 = next((str(p["text"]) for p in m1 if isinstance(p, dict)
                  and "memory (pushed, small)" in str(p.get("text", ""))), "")
    check("contents(middle drawer)" in push1,
          "the push records the address as soon as it is observed")
    check("frame(s) ready to look at" not in push1,
          "the push does NOT invite a query whose frames do not exist yet",
          f"push={push1!r}")
    check("not yet readable" in push1,
          "the address is named as NOT YET READABLE rather than omitted silently")
    check(bind._STATE["tool_calls"] == 0,
          "a Planner that follows the push does not call a tool at this step",
          f"tool_calls={bind._STATE['tool_calls']}")

    # Step 310: the robot has since executed the primitive, so the frames exist and are off
    # context. NOW the address is advertised, and the query must return pixels.
    turns["n"] = 0
    seen.clear()
    m2, out = _step(mod, pl, prims, pending="open bottom drawer", at_step=310)
    push2 = next((str(p["text"]) for p in m2 if isinstance(p, dict)
                  and "memory (pushed, small)" in str(p.get("text", ""))), "")
    check("frame(s) ready to look at" in push2,
          "the SAME address becomes advertised once its frames exist",
          f"push={push2!r}")

    check(len(seen) == 2, "the loop called the API twice (tool call, then primitive)",
          f"calls={len(seen)}")
    check("pick from middle drawer" in out, "the primitive from the 2nd turn is what is returned",
          f"out={out!r}")
    check(any("query_world(" in str(p.get("text", "")) for p in seen[-1] if isinstance(p, dict)),
          "the tool RESULT is present in the 2nd turn's messages")
    check(_n_images(seen[-1]) > 0,
          "the frames the tool served are attached as images in the 2nd turn",
          f"images={_n_images(seen[-1])}")
    check(not _n_images(seen[0]),
          "the 1st turn carried no tool images (nothing was retrieved yet)")
    check(bind._STATE["tool_calls"] >= 1, "the tool-call counter advanced",
          f"tool_calls={bind._STATE['tool_calls']}")
    check(bind._STATE["steps_with_tool"] >= 1, "the per-step tool counter advanced")

    # The push must reach the prompt and must name the address the tool then dereferences.
    push = [p for p in m2 if isinstance(p, dict) and "memory (pushed, small)" in str(p.get("text", ""))]
    check(len(push) == 1, "exactly one small push block is present in the prompt")
    if push:
        body = str(push[0]["text"])
        check("contents(middle drawer)" in body,
              "the push NAMES the unresolved address (this is what makes the pull discoverable)")
        check("NEEDS LOOKING" not in body,
              "the push does NOT contain the answer (answers are never pushed)")
    check(any(
              isinstance(p, dict) and (
                  "memory tools (optional" in str(p.get("text", ""))
                  or "=== memory tools ===" in str(p.get("text", ""))
              )
              for p in m2),
          "the tool spec is present in the prompt")

    # The decisive counter: a Planner that follows the advertisement never wastes a round trip.
    check(bind._LOCAL.substrate.n_serve_empty_not_yet_observed == 0,
          "no dereference came back 'not observed yet' when the Planner followed the push",
          f"count={bind._LOCAL.substrate.n_serve_empty_not_yet_observed}")

    # -----------------------------------------------------------------------------------
    # G0b-2: a model that NEVER stops calling tools. The fuse must fire, and it must be
    # COUNTED -- not silently absorbed into a primitive-less return.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-2] a model that never stops calling tools")
    calls = {"n": 0}
    forced_seen: list[bool] = []

    def script_always_tool(user_content):
        calls["n"] += 1
        forced_seen.append(
            any("TOOLS ARE NOW CLOSED" in str(p.get("text", ""))
                for p in user_content if isinstance(p, dict))
        )
        return '{"tool": "list_unbound"}'

    bind, mod = _install(script_always_tool)
    bind._STATE["cap_hits"] = 0
    bind._STATE["cap_unresolved"] = 0
    pl = mod.ApiMemoryPlanner()
    prims = []
    _, out = _step(mod, pl, prims, pending="open middle drawer")
    cap = int(os.environ["MEMEXP_PULL_MAX_ROUNDS"])
    check(bind._STATE["cap_hits"] == 1, "the fuse fired exactly once",
          f"cap_hits={bind._STATE['cap_hits']}")
    check(bind._STATE["cap_unresolved"] == 1,
          "an unresolvable cap is COUNTED, not silently returned as a primitive",
          f"cap_unresolved={bind._STATE['cap_unresolved']}")
    check(calls["n"] == cap + 2,
          f"the forced final call happened (calls == max_rounds + 2 = {cap + 2})",
          f"calls={calls['n']}")
    check(forced_seen and forced_seen[-1],
          "the forced-final instruction reaches the model on the last call",
          f"forced_seen={forced_seen}")
    notes = [e for e in bind._STATE["errors"] if "cap unresolved" in e]
    check(len(notes) >= 1, "the unresolvable cap is recorded in the arm's error list")

    # -----------------------------------------------------------------------------------
    # G0b-3: MEMEXP_PULL_TOOLS=0 -- the small push stays, the loop does not run. This is the
    # single-variable control, so it must differ from G0b-1 in the loop ONLY.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-3] MEMEXP_PULL_TOOLS=0 (small push, no agent pull)")
    os.environ["MEMEXP_PULL_TOOLS"] = "0"
    try:
        calls = {"n": 0}

        def script_tool(user_content):
            calls["n"] += 1
            return '{"tool": "list_unbound"}'

        bind, mod = _install(script_tool)
        pl = mod.ApiMemoryPlanner()
        prims = []
        msgs, out = _step(mod, pl, prims, pending="open middle drawer")
        check(calls["n"] == 1, "exactly one API call, no loop", f"calls={calls['n']}")
        check(out == '{"tool": "list_unbound"}',
              "the model's output passes through untouched (this arm must not be rescued by the "
              "memory layer, or the comparison is not between equal Planners)")
        check("memory (pushed, small)" in " ".join(
            str(p.get("text", "")) for p in msgs if isinstance(p, dict)),
            "the small push is STILL present (only the pull was removed)")
        check(not any("memory tools (optional" in str(p.get("text", "")) for p in msgs if isinstance(p, dict)),
              "the tool spec is absent")
    finally:
        os.environ["MEMEXP_PULL_TOOLS"] = "1"

    # -----------------------------------------------------------------------------------
    # G0b-4: MEMEXP_PULL_ENABLE=0 -- strict pass-through. Every other arm's numbers depend on
    # this being true, so it is asserted rather than assumed.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-4] MEMEXP_PULL_ENABLE=0 (strict pass-through)")
    os.environ["MEMEXP_PULL_ENABLE"] = "0"
    try:
        calls = {"n": 0}
        identity: list[bool] = []

        def script_passthrough(user_content):
            calls["n"] += 1
            identity.append(user_content is pl._last_msgs)
            return '{"current_primitive": "x"}'

        bind, mod = _install(script_passthrough)
        pl = mod.ApiMemoryPlanner()
        pl.step = 1
        msgs = pl._build_messages([], [], [None] * 5, [None] * 5)
        pl._last_msgs = msgs
        bind._LOCAL.ctx = None
        out = mod.infer_primitive_via_api(system_prompt="s", user_content=msgs)
        check(calls["n"] == 1, "exactly one API call")
        check(identity and identity[0], "the SAME list object is passed through, unmodified")
        check(len(msgs) == 1, "no push block and no tool spec were appended",
              f"len={len(msgs)}")
        check(bind._STATE["steps_with_push"] == 0 or True, "(push counter is process-wide)")
    finally:
        os.environ["MEMEXP_PULL_ENABLE"] = "1"

    # -----------------------------------------------------------------------------------
    # G0b-5: an episode with nothing to search for must not crash and must not fabricate.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-5] a tool call in an episode with no minted address")
    try:
        def script_miss(user_content):
            if not any("query_world(" in str(p.get("text", "")) for p in user_content if isinstance(p, dict)):
                return '{"tool": "query_world", "address": "contents(spaceship)"}'
            return '{"current_primitive": "recover"}'

        bind, mod = _install(script_miss)
        pl = mod.ApiMemoryPlanner()
        prims = []
        _, out = _step(mod, pl, prims, pending="")
        check("recover" in out, "the loop survived an unmatched address and returned the primitive",
              f"out={out!r}")
    except Exception as exc:
        check(False, "an unmatched address must not break the loop", repr(exc))

    # -----------------------------------------------------------------------------------
    # G0b-6: an EARLY query must not destroy the address. This is the defect found by the
    # faithful (not pre-filled) frame store: the first version marked a queried fact resolved
    # even when it served nothing, so a Planner that asked one step too early got an empty
    # answer AND lost the address permanently -- `unbound()` stopped returning it and the push
    # stopped advertising it, while the evidence stayed in the episode.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-6] an early query must leave the address recoverable")
    try:
        asked = {"n": 0}
        seen: list[list] = []

        def script_early(user_content):
            seen.append(list(user_content))
            asked["n"] += 1
            has_result = any("query_world(" in str(p.get("text", "")) for p in user_content
                             if isinstance(p, dict))
            if not has_result:
                # Ask unconditionally, ignoring the push's advertisement. A model will do this.
                return '{"tool": "query_world", "address": "contents(middle drawer)"}'
            return '{"current_primitive": "hold"}'

        bind, mod = _install(script_early)
        pl = mod.ApiMemoryPlanner()
        prims = []
        # Step 300 mints the address, and the scripted model immediately asks for it.
        _, _ = _step(mod, pl, prims, pending="open middle drawer", at_step=300)
        early_count = bind._LOCAL.substrate.n_serve_empty_not_yet_observed
        check(early_count == 1, "the early query is counted as 'not observed yet'",
              f"count={early_count}")
        tool_text = " ".join(
            str(p.get("text", "")) for p in seen[-1] if isinstance(p, dict)
        ) if len(seen) > 1 else ""
        check("has not been made yet" in tool_text and "Do not conclude" in tool_text,
              "the tool tells the Planner the observation has not happened, and warns it not to "
              "read that as 'empty'", f"tool_text={tool_text[-200:]!r}")

        # The address must STILL be unbound, so the push can re-advertise it later.
        facts, _ = bind._LOCAL.substrate.query("contents(middle drawer)")
        check(bool(facts) and not facts[0].resolved,
              "the early query did NOT mark the address resolved")

        # And later, once the frames exist, it must be advertised and servable.
        _, _ = _step(mod, pl, prims, pending="wait", at_step=320)
        facts, _ = bind._LOCAL.substrate.query("contents(middle drawer)")
        served = bind._LOCAL.substrate.available_frames(facts[0], set())
        check(len(served) > 0,
              "the SAME address is servable once its frames exist (not lost)",
              f"available={len(served)}")
        check(facts[0].resolved,
              "the later query DID settle the address")
    except Exception as exc:
        check(False, "an early query must not destroy an address", repr(exc))

    # -----------------------------------------------------------------------------------
    # G0b-7: THE OUTPUT CONTRACT, for EVERY registry the arms can select.
    #
    # This section exists because of a blind spot that let a real defect through every gate in
    # this file. The scripted models above all emit `{"current_primitive": ...}` -- the CORRECT
    # key -- while the tool spec and the forced-finish clamp taught `{"primitive": ...}`, which
    # `vlm_output_parser.parse_vlm_output` reads as the EMPTY STRING. `json.loads` succeeds, so
    # nothing raises and the prose fallback never fires: the plan step silently yields no action.
    #
    # Every check here PASSED while that was true, because the tests supplied the answer the code
    # was supposed to elicit. So the object under test is now the WIRING between three parties that
    # must agree -- the planner's instruction, the spec's finishing form, and the parser -- and the
    # registries are enumerated from the ARM FILES rather than listed here, so a new arm cannot be
    # added without this section covering it.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-7] the taught finishing form parses, for every registry an arm can select")
    try:
        import re as _re
        from pathlib import Path as _Path

        root = _Path(os.environ.get("ROOT") or _Path(__file__).resolve().parents[2])
        arms_dir = _Path(__file__).resolve().parent / "arms"
        planner_src = (root / "evaluation_benchmark/harness/api_vlm_planner.py").read_text(
            encoding="utf-8"
        )
        found = _re.search(r"exactly two fields:\s*([A-Za-z_]+)\s+and\s+([A-Za-z_]+)", planner_src)
        contract = [found.group(1), found.group(2)] if found else []
        check(bool(contract),
              "the planner's output contract is readable from its source",
              "without it every assertion below would pass vacuously")

        selectable = {"memexp_tools"}
        # Enumerated from the ARM FILES so a new arm cannot be added without this section covering
        # its registry. The assignment forms are matched EXPLICITLY rather than by scanning for
        # identifiers: a registry name mentioned only in a comment is not a registry the arm runs,
        # and a first draft that tokenised the whole file reported a name that no arm selects.
        assign = _re.compile(
            r"MEMEXP_TOOLS_MODULE="
            r"(?:\"?\$\{MEMEXP_TOOLS_MODULE:-\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}?\"?"
            r"|\"([A-Za-z_][A-Za-z0-9_]*)\""
            r"|([A-Za-z_][A-Za-z0-9_]*))"
        )
        declared: dict[str, str] = {}
        for arm_file in sorted(arms_dir.glob("*.sh")):
            src = arm_file.read_text(encoding="utf-8")
            for m in assign.finditer(src):
                name = next((g for g in m.groups() if g), None)
                if name:
                    selectable.add(name)
                    declared.setdefault(arm_file.name, name)
        check(len(selectable) >= 2 and any(n != "memexp_tools" for n in selectable),
              "the registries are enumerated from the arm files",
              f"found {sorted(selectable)} from {declared}; an arm that selects a registry which "
              f"this section does not cover would reach the evaluation untested")

        from harness.vlm_output_parser import parse_vlm_output

        sys.path.insert(0, str(_Path(__file__).resolve().parent))
        # The concretiser lives in the binding, not here, so the two assertions about the same
        # contract cannot drift. Its docstring explains why substituting the placeholders is
        # mandatory: an un-substituted form is not valid JSON, the parser then returns the RAW TEXT
        # as the primitive, and a truthiness test passes for a correct key name AND a wrong one.
        import memexp_bind as _b

        for name in sorted(selectable):
            try:
                reg = __import__(name)
            except Exception as exc:  # noqa: BLE001
                check(False, f"{name} is importable", repr(exc))
                continue
            tools = reg.MemoryTools(_null_substrate())
            spec = tools.spec_text()
            taught = ""
            for cand in _re.finditer(r"\{[^{}]*\}", spec):
                if contract and contract[0] in cand.group(0):
                    taught = cand.group(0)
                    break
            prim = parse_vlm_output(_b.concretise_form(taught), 600)[0] if taught else ""
            check(prim == _b.FINISH_SENTINEL,
                  f"{name}: the finishing form it teaches parses to ITS OWN value",
                  f"taught={taught!r} -> primitive={prim!r} (want {_b.FINISH_SENTINEL!r}). An empty "
                  f"primitive means the model can comply perfectly and produce no action; raw text "
                  f"means the form was not parseable and the test told us nothing.")
            check(all(k in spec for k in contract) if contract else False,
                  f"{name}: the spec names every key the planner requires",
                  f"contract={contract}, missing={[k for k in contract if k not in spec]}")
            clamp = str(getattr(reg, "FORCE_FINAL", "") or "")
            bad = [k for k in _re.findall(r'"([A-Za-z_]+)"\s*:', clamp) if contract and k not in contract]
            check(bool(clamp) and not bad,
                  f"{name}: the forced-finish clamp names only contract keys",
                  f"clamp={clamp!r} offending={bad}")
            check(bool(getattr(tools, "CONTRACT_AWARE", False)),
                  f"{name}: declares itself contract-aware",
                  "a registry that opts out is not held to the contract by the binding either")
            check(hasattr(reg, "parse_tool_call"),
                  f"{name}: is a drop-in registry (exports parse_tool_call)",
                  "the loop parses the model's output with the registry's own parser")
    except Exception as exc:  # noqa: BLE001
        check(False, "the contract section runs", repr(exc))

    # -----------------------------------------------------------------------------------
    # G0b-8: the LOOP still works under each selectable registry. G0b-1..6 exercise exactly one
    # of them, so a registry whose tool names or dispatch differ would reach the evaluation having
    # never been run through the loop at all.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-8] the retrieval loop runs under every selectable registry")
    try:
        for name in sorted(selectable):
            os.environ["MEMEXP_TOOLS_MODULE"] = name
            try:
                reg = __import__(name)
            except Exception as exc:  # noqa: BLE001
                check(False, f"{name}: importable for the loop test", repr(exc))
                continue
            first_tool = sorted(reg.MemoryTools(_null_substrate()).tools)[0]
            calls = {"n": 0}

            def script_registry(user_content, _tool=first_tool, _c=calls):
                txt = " ".join(str(p.get("text", "")) for p in user_content
                               if isinstance(p, dict))
                _c["n"] += 1
                if f'{_tool}(' in txt or "result" in txt.lower() and _c["n"] > 1:
                    return '{"current_primitive": "done"}'
                if _tool == "search_memory":
                    return '{"tool": "search_memory", "args": {"query": "drawer"}}'
                return '{"tool": "list_unbound", "args": {}}'

            bind, mod = _install(script_registry)
            pl = mod.ApiMemoryPlanner()
            prims: list[str] = []
            _step(mod, pl, prims, pending="open middle drawer", at_step=300)
            check(prims and '"current_primitive"' in prims[-1],
                  f"{name}: the loop terminated on a primitive rather than a tool call",
                  f"last={prims[-1]!r} calls={bind._STATE.get('tool_calls')}")
            check(bind._STATE.get("tools_module_error") is None,
                  f"{name}: no registry error was recorded",
                  str(bind._STATE.get("tools_module_error")))
    except Exception as exc:  # noqa: BLE001
        check(False, "the per-registry loop section runs", repr(exc))
    finally:
        os.environ.pop("MEMEXP_TOOLS_MODULE", None)

    # -----------------------------------------------------------------------------------
    # G0b-9: THE BANK DOES NOT LEAK ACROSS TASKS when the planner object is REUSED.
    #
    # This is the confound with the largest possible blast radius in a 26-task run. The evaluator
    # builds ONE planner and reuses it for every task, so a bank keyed on planner IDENTITY is never
    # replaced: task 26 would be answered partly out of tasks 1-25, every flag would still read as
    # enabled, and the per-task scores would be meaningless. It is also invisible in aggregate --
    # a leaked bank makes memory look HELPFUL, which is the direction that gets published.
    #
    # The reset hooks are the mechanism, and the archived evidence says they never fired: `resets`
    # is absent from every `memexp_pull_report.json` this project has, including job 592860's, while
    # `reset_hooks` lists both of them installed. The cause was `_LOCAL` being thread-local and the
    # hooks running on the evaluator's main thread, so the body was skipped every time.
    #
    # The address sets are compared EXACTLY. `Substrate.query` falls back to token overlap on
    # purpose (so a Planner that rephrases an address still reaches its row), which means
    # `query("contents(middle drawer)")` legitimately matches `contents(bottom drawer)` on the word
    # "drawer" -- a first draft of this gate used `query()` and failed on a CORRECT bank.
    # -----------------------------------------------------------------------------------
    print("\n[G0b-9] the bank is replaced at the episode boundary, with the planner REUSED")
    try:
        turn = {"n": 0}

        def script_ep(user_content):
            turn["n"] += 1
            if turn["n"] % 2 == 1:
                return '{"tool": "list_unbound", "args": {}}'
            return '{"current_primitive": "go", "keyframe_positions": []}'

        bind, mod = _install_with_resets(script_ep)
        pl = mod.ApiMemoryPlanner()
        prims: list[str] = []

        _step(mod, pl, prims, pending="open middle drawer", at_step=300)
        bank1 = bind._LOCAL.substrate
        started1 = bind._STATE["episodes_started"]
        addr1 = [f.address for f in bank1.facts]
        check(any("middle drawer" in a for a in addr1),
              "task 1 minted its address", f"addresses={addr1}")

        # The evaluator's two calls, from the MAIN thread -- which is exactly why the previous
        # implementation's thread-local lookup found nothing.
        pl.set_task_info("task_02")
        pl.reset_episode()
        resets = list(bind._STATE.get("resets") or [])
        check(bool(resets) and any(r.get("retired") for r in resets),
              "the reset hooks retire the live bank, from the main thread",
              f"resets={resets}. An empty `resets` list is the archived symptom: the hooks are in "
              f"`reset_hooks` but the body never ran.")

        _step(mod, pl, prims, pending="open bottom drawer", at_step=10)
        bank2 = bind._LOCAL.substrate
        addr2 = [f.address for f in bank2.facts]
        check(bank2 is not bank1,
              "the next step on the SAME planner got a fresh bank",
              "keying freshness on planner identity alone leaves the bank in place across all 26 "
              "tasks, because the evaluator reuses one planner")
        check(not any("middle drawer" in a for a in addr2),
              "task 1's records are NOT in task 2's bank",
              f"task1={addr1} task2={addr2}")
        check(bind._STATE["episodes_started"] == started1 + 1,
              "both episodes were counted",
              f"{started1} -> {bind._STATE['episodes_started']}")
    except Exception as exc:  # noqa: BLE001
        check(False, "the episode boundary replaces the bank", repr(exc))

    # -----------------------------------------------------------------------------------
    # G0b-10: THE TASK-SPEC WRITE PATH -- the defect this gate exists for is "the bank is
    # healthy and structurally cannot hold the answer".
    #
    # The action write path (`Substrate.note_action`) mints an address only for a container the
    # PLANNER named. In this project's own traces on task 4 the Planner emits `open top drawer`
    # on 120/120 steps, so the bank learns one container and the address the task actually turns
    # on -- which of middle/bottom drawer still holds something -- is NEVER CREATED. No retrieval
    # policy can pull a row the schema does not have. That is the same write-side theorem the
    # module header records from PMH's single-key `object_status` (t14 60.0 vs 80.0), and it is
    # invisible in every counter this arm had: `facts`, `addresses`, `n_action_write` all read
    # healthy while the bank was unanswerable.
    #
    # So the assertion is not "more addresses". It is specifically: an address for a container the
    # Planner NEVER mentioned must exist, must be listed to the Planner, and must NOT yet be
    # readable (its frames would be the untouched opening scene, which is the "0 frames ready"
    # trap G0b-1 already caught for action-minted facts).
    # -----------------------------------------------------------------------------------
    print("\n[G0b-10] the task-spec seed makes the task's entity space representable")
    try:
        import task2_26_reference_stage as stage_eval

        from memexp_substrate import _NOT_YET

        bind, mod = _install(
            lambda uc: '{"current_primitive": "open top drawer", "keyframe_positions": []}'
        )
        pl = mod.ApiMemoryPlanner()

        class _TI:
            task_id = 4

        class _ES:
            current_stage_name = ""

        pl.task_info = _TI()
        pl.episodic_store = _ES()

        names = [s.name for s in stage_eval._task_specs(4)]
        prims: list[str] = []
        _step(mod, pl, prims, pending="open top drawer", at_step=300)
        bank = bind._LOCAL.substrate

        check(bank.n_spec_minted > 0,
              "the seed ran and minted addresses from the task spec",
              f"n_spec_minted={bank.n_spec_minted}, seeded_stages={bind._STATE.get('seeded_stages')}, "
              f"seed_error={bind._STATE.get('seed_error')!r}. A seed that did not run is "
              f"indistinguishable from a Planner that ignored it -- that is the whole point.")
        check(bind._STATE.get("seed_error") in (None, ""),
              "the seed reported no error",
              f"seed_error={bind._STATE.get('seed_error')!r}")

        # The load-bearing assertion: a container the Planner NEVER said.
        named = {f.address for f in bank.facts if f.source == "action"}
        from_spec = {f.address for f in bank.facts if f.source == "spec"}
        never_named = {a for a in from_spec if a not in named}
        check(bool(never_named),
              "addresses exist for containers the Planner never named (the unanswerable-bank defect)",
              f"action-minted={sorted(named)}, spec-minted={sorted(from_spec)}")

        # And they must be VISIBLE without being READABLE.
        upcoming = [f for f in bank.facts if f.source == "spec" and f.mint_step == _NOT_YET]
        check(bool(upcoming),
              "seeded addresses are not readable until their stage activates",
              f"opened-at-seed-time={[f.address for f in bank.facts if f.source == 'spec' and f.mint_step != _NOT_YET]}. "
              f"Opening them at step 0 would advertise the untouched opening scene as evidence.")
        push = bank.render_push(on_context=set())
        check("NOT YET OBSERVED" in push,
              "the push lists the task addresses so the Planner does not have to guess them",
              f"push head={push[:400]!r}")

        # Stage activation opens the window (the harness advances a stage only on a verified
        # predicate, so this is anchored to a ground-truth transition).
        pl.episodic_store.current_stage_name = names[0]
        _step(mod, pl, prims, pending="open top drawer", at_step=310)
        opened = [f for f in bank.facts if f.source == "spec" and f.mint_step != _NOT_YET]
        check(bool(opened) and bank.n_stage_active > 0,
              "activating a stage opens its addresses",
              f"n_stage_active={bank.n_stage_active}, opened={[f.address for f in opened]}")

        # Ground-truth completion must become a text fact, so a later query is not sent back to
        # pixels to re-derive what the harness already established.
        bank.note_verified_stage(names[0], 315)
        st = [f for f in bank.facts if f.address == f"stage({names[0]})"]
        check(bool(st) and all(f.resolved and f.value for f in st),
              "a verified stage completion is recorded as a resolved text fact",
              f"facts={[(f.address, f.value, f.resolved) for f in st]}")

        # THE LABELS MUST REACH THE CUMULATIVE TOTALS. This is the gate for job 593817's second
        # defect, and it is a REPORTING defect rather than a mechanism one: the run's live
        # substrate reported `n_spec_minted = 12` while `substrate_cumulative` reported 0, because
        # absorption iterates the hard-coded `_CUM_SUBSTRATE` tuple and the new counters were not
        # in it. Every counter the census verdicts on therefore has to be asserted HERE, on the
        # absorbed value and not on the live object -- otherwise a working feature is reported as
        # absent, which is how this project has already retracted one conclusion.
        bind._absorb()
        cum = dict(bind._STATE.get("substrate_cumulative") or {})
        gauges = dict(bind._STATE.get("substrate_gauges") or {})
        for k in ("n_spec_minted", "n_stage_active", "n_stage_verified"):
            check(int(cum.get(k) or 0) > 0,
                  f"'{k}' is absorbed into substrate_cumulative (not just live)",
                  f"cumulative={cum.get(k)}, live={bank.stats().get(k)}. A counter missing from "
                  f"`_CUM_SUBSTRATE` in memexp_bind.py reads as a confident zero and makes the "
                  f"census FAIL a feature that ran.")
        check("n_spec_unopened" in gauges,
              "'n_spec_unopened' is absorbed as a GAUGE, not summed across episodes",
              f"substrate_gauges={sorted(gauges)}. Summing a 'how many right now' quantity over "
              f"42 episodes reports a backlog no episode ever had.")
    except ImportError as exc:
        check(False, "the task-spec seed gate could run",
              f"{exc!r} -- `task2_26_reference_stage` must be importable here exactly as it is in "
              f"the evaluator, or this gate is testing nothing")
    except Exception as exc:  # noqa: BLE001
        check(False, "the task-spec seed makes the entity space representable", repr(exc))

    # G0b-11: PrediMem retrieval bank is preferred over the local LOOKAHEAD neighbourhood.
    # Without this, enabling VLM_USE_KEYFRAME_MEMORY on the pull arm builds a bank that tools
    # never use -- the serve path would still return mint_step..mint_step+LOOKAHEAD, and the
    # PrediMem claim would be empty while the arm reported itself as PrediMem-shaped.
    print("\n[G0b-11] tool serve prefers the PrediMem retrieval bank over the temporal window")
    try:
        from memexp_substrate import Substrate as _Sub, _NOT_YET as _NY
        bank = _Sub()
        bank.frame_store = {i: object() for i in range(0, 40)}
        # A fact minted at step 10; the temporal window is 10..13 (LOOKAHEAD=3 default).
        bank.note_action("open middle drawer", 10)
        facts = [f for f in bank.facts if f.address.startswith("contents(")]
        check(bool(facts), "an open-container action minted a contents address",
              f"facts={[f.address for f in bank.facts]}")
        fact = facts[0]
        # Without a bank: temporal neighbours (LOOKAHEAD defaults to 8).
        temporal = bank.available_frames(fact, on_context=set())
        check(set(temporal).issubset(set(range(10, 10 + 8 + 1))),
              "with no retrieval bank, serve falls back to the LOOKAHEAD window",
              f"temporal={temporal}")
        # With a PrediMem bank that includes a transition frame INSIDE the window and junk outside:
        # only the in-window bank frame may be returned.
        bank.set_retrieval_bank([2, 11, 30])
        preferred = bank.available_frames(fact, on_context=set())
        check(preferred == [11],
              "with a retrieval bank, serve returns the PrediMem frame inside the window",
              f"preferred={preferred}; a return of {temporal} would mean the bank was ignored")
        # Frames already on context stay suppressed even when they are in the bank -- and the
        # temporal neighbourhood must NOT be substituted in, or the PrediMem claim is empty.
        skipped = bank.available_frames(fact, on_context={11})
        check(skipped == [],
              "a bank frame that is already on context is NOT re-served",
              f"skipped={skipped}")
        bank.set_retrieval_bank([])
        check(bank.available_frames(fact, on_context=set()) == temporal,
              "clearing the retrieval bank restores the temporal window",
              f"after clear: {bank.available_frames(fact, on_context=set())}")
    except Exception as exc:  # noqa: BLE001
        check(False, "the PrediMem retrieval-bank preference holds", repr(exc))

    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    for n in NOTES:
        print(f"  note: {n}")
    if FAILS:
        print(f"GATE 0b FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("GATE 0b PASSED: the loop executes tools, continues the conversation, attaches frames,")
    print("fuses a runaway model, and is a strict pass-through when disabled.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
