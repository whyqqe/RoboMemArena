"""ECHO: test causal boundaries and the actual late-import binding contract."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

from .core import EchoState, planner_constraint, primitive_reason, sanitize_output


def run() -> None:
    s = EchoState()
    s.anticipate("lift", 5, {5: "frame"})
    s.delivered("Lift the bottle", 6)
    s.delivered("Lift the bottle", 7)
    assert s.commitment and s.commitment.started == 6 and s.commitment.last_seen == 7
    assert s.commitment.status == "DELIVERED" and s.n_delivered == 2
    text, frames = s.decision_package({5: "frame"})
    assert frames == [5] and "not confirmed" in text
    s.anticipate("pour", 8, {8: "frame"})
    assert s.commitment.status == "AMBIGUOUS" and not s.verified
    for source in ("stage_done", "scorer", "planner"):
        try:
            s.independent_verification("lift", True, source)
            raise AssertionError(f"accepted forbidden source {source}")
        except ValueError:
            pass
    s.independent_verification("lift", True, "visual_verifier")
    assert "lift" in s.verified
    assert not s.decision_package({5: "frame"})[1], "old stage must not leak into next stage"
    s.anticipate("pour", 260, {260: "frame"})
    assert not s.decision_package({8: "frame"})[1], "expired evidence was served"

    assert primitive_reason('Pick up the chocolate and place it in the frypan.', 0)
    assert primitive_reason('Pour the tomato sauce.', 0)
    assert primitive_reason('Pour tomato sauce again.', 1)
    assert primitive_reason('Grasp the tomato sauce bottle and lift it.', 0) is None
    replaced, reason = sanitize_output('{"current_primitive":"Pick up the chocolate","keyframe_positions":[]}', 0)
    assert reason and 'tomato sauce bottle' in json.loads(replaced)['current_primitive']
    assert sanitize_output('{broken', 0) == ('{broken', None)
    assert 'second time' in planner_constraint(2)

    # Sitecustomize runs before harness appears on sys.path; ensure genuine late import.
    os.environ["MEMEXP_ECHO"] = "1"
    saved = {k: v for k, v in sys.modules.items() if k == "harness" or k.startswith("harness.")}
    for key in saved:
        sys.modules.pop(key)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MEMEXP_ECHO_REPORT"] = str(Path(tmp) / "receipt.json")
        from . import bind
        bind._INSTALLED = False
        bind._BOUND.clear()
        bind._ACTIVE = None
        bind.install()
        assert not bind._BOUND
        pkg = Path(tmp) / "harness"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "api_vlm_planner.py").write_text(
            "class ApiMemoryPlanner:\n"
            " def __init__(self):\n"
            "  self.step=1; self.frame_store_main={1:object()}; "
            "self._current_subtask='lift'; self.task_info=type('T',(),{'task_id':8})()\n"
            " def _build_messages(self,*args,**kwargs): return [{'type':'text','text':'base'}]\n"
            " def reset_episode(self,*args,**kwargs): self.step=0\n"
            "infer_primitive_via_api=lambda **kwargs: 'unchanged'\n")
        (pkg / "controller.py").write_text(
            "class HarnessController:\n"
            " def override_vla_prompt(self,prompt,*args,**kwargs): return prompt\n"
            " def on_episode_end(self,*args,**kwargs): return None\n")
        sys.path.insert(0, tmp)
        try:
            ap = importlib.import_module("harness.api_vlm_planner")
            ctl = importlib.import_module("harness.controller")
            assert bind._BOUND == bind._TARGETS, bind._BOUND
            assert ap.infer_primitive_via_api(system_prompt="", user_content=[]) == "unchanged"
            assert ctl.HarnessController().override_vla_prompt("Lift the bottle", step=1) == "Lift the bottle"
            planner = ap.ApiMemoryPlanner()
            first = planner._build_messages([], [], [], [])
            assert first[0] == {"type": "text", "text": "base"}
            assert any("grasp and lift" in m.get("text", "") for m in first)
            state = bind._state(planner)
            import threading
            observed = []
            thread = threading.Thread(target=lambda: observed.append(bind._state()))
            thread.start()
            thread.join()
            assert observed == [state], "controller and planner must share one episode state"
            calls = []
            def fake_api(**kw):
                calls.append(kw)
                primitive = "Pick up the chocolate" if len(calls) == 1 else "Grasp the tomato sauce bottle and lift it."
                return json.dumps({"current_primitive": primitive, "keyframe_positions": []})
            # The wrapper closes over the original API; the mock module's original is
            # replaced here only by rebuilding the wrapper for this test fixture.
            old_api = ap.infer_primitive_via_api
            original = old_api.__closure__
            assert original is not None
            api_cell = next(c for c in original if callable(c.cell_contents) and c.cell_contents is not old_api)
            previous = api_cell.cell_contents
            api_cell.cell_contents = fake_api
            try:
                output = ap.infer_primitive_via_api(system_prompt="", user_content=first)
                assert len(calls) == 2 and "tomato sauce bottle" in json.loads(output)["current_primitive"]
                assert state.n_retries == 1
            finally:
                api_cell.cell_contents = previous
            assert state.n_saved == 1 and state.n_plans == 1
            planner.step = 30
            planner._consecutive_same_subtask = 3
            planner.frame_store_main[30] = object()
            msgs = planner._build_messages([], [], [], [])
            assert any("ECHO prospective" in str(m) for m in msgs), msgs
            assert any(m.get("type") == "image" for m in msgs)
            ctl.HarnessController().on_episode_end(stage_done={"lift": True})
            assert not state.verified, "scorer leaked into online ledger"
            receipt = json.loads(Path(f"{os.environ['MEMEXP_ECHO_REPORT']}.{os.getpid()}").read_text())
            assert set(receipt["bound"]) == bind._TARGETS and receipt["episode"]["n_plans"] == 2
            planner.reset_episode()
            assert bind._state(planner) is not state and not bind._state(planner).evidence
        finally:
            sys.path.remove(tmp)
            for key in list(sys.modules):
                if key == "harness" or key.startswith("harness."):
                    sys.modules.pop(key)
            sys.modules.update(saved)
            os.environ.pop("MEMEXP_ECHO_REPORT", None)
            os.environ.pop("MEMEXP_ECHO", None)
    print("ECHO selftest PASS: prospective evidence, no oracle credit, late hook, identity, reset, receipt")


if __name__ == "__main__":
    run()
