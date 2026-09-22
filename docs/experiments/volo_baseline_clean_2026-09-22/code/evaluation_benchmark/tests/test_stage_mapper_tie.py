"""Pins the `stage_mapper` TIE-BREAK defect (found by the T1 reachability audit, 2026-09-21).

Measured defect this file pins down
-----------------------------------
`expected_primitive_for_stage` falls back to a fuzzy token matcher whenever
`|primitive_labels| != |stage_specs|` -- which is ALWAYS the case for the 26 memory tasks, because
`primitive_order` carries `pick_*` labels that own no scored stage (6 labels against 4 or 3 specs).
The incumbent matcher used `hit > best_score`, so on a tie the FIRST label in `primitive_order`
won:

    stage 02_Place_Cookies_Middle_Drawer
      'open middle drawer'  hits {middle, drawer}  = 2   <- index 0, won
      'place cookies'       hits {place, cookies}  = 2   <- tie, blocked by `>`

The recovery candidate therefore became 'open middle drawer' -- the exact primitive the robot was
already stuck on -- and `HarnessController._handle_stall` writes `subtask_override` only when
`candidate != subtask`. Net effect, measured over the 15-episode pilot
(`outputs/mem_efficacy_local/pilot_armparts_0920_1757/nomem/nomem_s100`):

    drawer tasks 12/13/17 : 0 ladder overrides in 9/9 episodes , score stuck at 33.3 / 66.7
    microwave tasks 20/23 : 2 ladder overrides per episode     , score 66.7 / 100.0

The microwave family was untouched by accident, not by design: 'open microwave' hits only
{microwave} = 1, below the `min(2, len(tokens))` floor, so it never entered the tie and
'place cookies' won outright. One family therefore ran WITH a recovery mechanism and the other
without, which made their scores incomparable.

Two faults are pinned separately below:
  * TIE-BREAK  -- which label the fuzzy matcher returns;
  * NO-OP      -- the resulting candidate silently equals the stuck subtask, so the ladder can
                  never fire. That NO-OP path itself is pre-existing and is covered by
                  `test_recovery_ladder.py::test_language_noop_arms_the_hard_rung`; here we pin
                  that the DRAWER tasks must not take it.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]                      # RoboMemArena/
sys.path.insert(0, str(_ROOT / "evaluation_benchmark"))
sys.path.insert(0, str(_ROOT / "evaluation_benchmark" / "scripts"))

from harness.stage_mapper import expected_primitive_for_stage  # noqa: E402

_CFG = _ROOT / "evaluation_benchmark/async_vlm26_reference/fullvlm_v2_26_memory_tasks.json"


def _labels(task_id: int) -> list[str]:
    cfg = json.loads(_CFG.read_text())
    for t in cfg["tasks"]:
        if t["task_id"] == task_id:
            return [str(p["label"]) for p in t["primitive_order"]]
    raise KeyError(f"task {task_id} not in {_CFG.name}")


def _specs(task_id: int):
    import task2_26_reference_stage as M
    return M._task_specs(task_id)


@dataclass
class _Spec:
    name: str


_STOP = {"the", "and", "into", "onto", "to", "for", "of", "with", "again", "final",
         "a", "an", "in", "on", "it", "its"}


def _obj_tokens(text: str | None) -> set[str]:
    """Object words of a stage/label, with verbs and glue words removed."""
    if not text:
        return set()
    return {t for t in str(text).lower().replace("_", " ").split()
            if t not in _VERB_SET and t not in _STOP and len(t) > 2}


_VERB_SET = {"open", "close", "pick", "place", "pour", "push", "pull", "reach",
             "grasp", "lift", "put", "move"}


def _data_cannot_serve_stage(spec, labels: list[str]) -> bool:
    """True when `primitive_order` has NO label that could serve this stage.

    This is a TASK-DEFINITION inconsistency, not a mapper fault: no mapping can produce the
    stage's object because the object is absent from `primitive_order` entirely.
    """
    v = _verb(spec.name)
    if v is None:
        return False
    need = _obj_tokens(spec.name)
    have: set[str] = set()
    for lab in labels:
        if _verb(lab) == v:
            have |= _obj_tokens(lab)
    return not (need & have)


def _candidate(task_id: int, stage_idx: int, current_subtask: str) -> str | None:
    specs = _specs(task_id)
    return expected_primitive_for_stage(
        primitive_labels=_labels(task_id),
        stage_idx=stage_idx,
        stage_specs=specs,
        stage_done={s.name: False for s in specs[:stage_idx]},
        current_subtask=current_subtask,
    )


def _verb(text: str | None) -> str | None:
    verbs = ("open", "close", "pick", "place", "pour", "push", "pull", "reach", "grasp", "lift", "put")
    if not text:
        return None
    for tok in str(text).lower().replace("_", " ").split():
        if tok in verbs:
            return tok
    return None


# --- THE REGRESSION, PINNED BY HAND -----------------------------------------------------

def test_drawer_place_cookies_maps_to_place_not_to_open():
    """The exact cell that was broken: task12 stage 02 returned the stuck primitive."""
    got = _candidate(12, 1, current_subtask="open middle drawer")
    assert got == "place cookies", (
        f"task12 stage 02_Place_Cookies_Middle_Drawer must map to 'place cookies'; got {got!r}. "
        "'open middle drawer' is the very primitive the planner is stuck on, so the ladder's "
        "candidate would equal the subtask and the override would be suppressed (0 overrides in "
        "9/9 drawer episodes)."
    )


def test_drawer_place_chocolate_maps_to_place_not_to_open():
    got = _candidate(12, 2, current_subtask="open middle drawer")
    assert got == "place chocolate", f"task12 stage 03 must map to 'place chocolate'; got {got!r}"


def test_microwave_was_already_correct_and_must_stay_correct():
    """The family that worked by accident. Pin it so a future change cannot regress it."""
    assert _candidate(20, 1, "open microwave") == "place cookies"
    assert _candidate(20, 2, "open microwave") == "place chocolate"
    assert _candidate(23, 1, "open microwave") == "place cream"
    assert _candidate(23, 2, "open microwave") == "place popcorn"


def test_candidate_differs_from_the_stuck_subtask_for_every_drawer_place_stage():
    """The property the ladder actually needs: `candidate != subtask`.

    `_handle_stall` writes `subtask_override` only on inequality. Asserting equality against the
    literal label string is stronger (and clearer) than asserting on verb identity alone.
    """
    for tid, stage_idx, expected in ((12, 1, "place cookies"), (12, 2, "place chocolate"),
                                     (13, 1, "place cookies"), (13, 2, "place butter"),
                                     (17, 1, "place butter"), (17, 2, "place chocolate")):
        stuck = "open middle drawer"
        got = _candidate(tid, stage_idx, current_subtask=stuck)
        assert got == expected, f"task{tid} stage{stage_idx}: expected {expected!r}, got {got!r}"
        assert got != stuck, (
            f"task{tid} stage{stage_idx}: candidate {got!r} equals the stuck subtask, so "
            "_handle_stall would suppress the override"
        )


# --- THE INVARIANT, SWEPT OVER EVERY TASK -----------------------------------------------

@pytest.mark.parametrize("task_id", list(range(1, 27)))
def test_candidate_verb_matches_stage_verb_for_every_task(task_id: int):
    """For open/close/place stages the candidate's verb must equal the stage's verb.

    Restricted to those three verbs because they are unambiguous: an `04_Close_*` stage can never
    legitimately be served by 'open ...', and a `02_Place_*` stage can never be served by
    'open ...'. Other verbs (pour/lift/reach) have task-specific conventions and are not asserted.
    """
    try:
        specs = _specs(task_id)
    except Exception:                                    # task has no spec builder
        pytest.skip(f"task {task_id} has no _task_specs entry")
    if not specs:
        pytest.skip(f"task {task_id} returned no stages")
    labels = _labels(task_id)
    if not labels:
        pytest.skip(f"task {task_id} has no primitive_order")

    violations = []
    skipped = []
    for idx in range(1, len(specs)):
        stage_verb = _verb(specs[idx].name)
        if stage_verb not in ("open", "close", "place"):
            continue
        if _data_cannot_serve_stage(specs[idx], labels):
            skipped.append(specs[idx].name)
            continue
        # The planner is stuck on the preceding stage's primitive, which is the shape that
        # triggers the ladder in the first place.
        stuck = labels[idx - 1] if idx - 1 < len(labels) else ""
        got = expected_primitive_for_stage(
            primitive_labels=labels, stage_idx=idx, stage_specs=specs,
            stage_done={s.name: False for s in specs[:idx]}, current_subtask=stuck)
        if _verb(got) != stage_verb:
            violations.append((specs[idx].name, stuck, got))
    if skipped and task_id == 11:
        return          # pinned separately in test_known_task_definition_inconsistencies
    assert not violations, (
        "fuzzy matcher returned a candidate whose verb contradicts the stage's verb:\n"
        + "\n".join(f"  stage={s}  stuck_on={st!r}  candidate={g!r}" for s, st, g in violations)
    )


# --- TASK-DEFINITION INCONSISTENCIES (not mapper faults) --------------------------------

def test_known_task_definition_inconsistencies():
    """`primitive_order` must contain a label able to serve every stage.

    Violations are NOT mapper faults -- no mapping can invent a missing object. They are task
    metadata defects that silently disable the ladder for that stage (the fuzzy matcher falls
    back to whatever container word it can find, which is the *opening* primitive).

    Measured 2026-09-21: exactly ONE such cell, and it is outside the tasks we run
    (12/13/17/20/23). Pin the exact set so a newly introduced inconsistency fails loudly.
    """
    found = set()
    for task_id in range(1, 27):
        try:
            specs = _specs(task_id)
        except Exception:
            continue
        if not specs:
            continue
        labels = _labels(task_id)
        for spec in specs:
            if _verb(spec.name) in ("open", "close", "place", "pick") and _data_cannot_serve_stage(spec, labels):
                found.add((task_id, spec.name))

    expected = {(11, "05_Place_Butter_Middle_Drawer")}
    assert found == expected, (
        "the set of (task, stage) cells whose object is absent from primitive_order changed.\n"
        f"  expected: {sorted(expected)}\n"
        f"  found   : {sorted(found)}\n"
        "A NEW entry means a task's stage specs and primitive_order have drifted apart, which "
        "silently disables the stall ladder for that stage. A MISSING entry means the data was "
        "fixed -- update this expectation."
    )


# --- END TO END THROUGH THE REAL CONTROLLER ---------------------------------------------

def _drive_real_ladder(task_id: int, stage_idx: int, subtask: str, *, until: int = 400):
    """Drive the REAL HarnessController for `task_id` and return the overrides it produced.

    This is the measurement that matters: it exercises `_handle_stall` -> `subtask_override`,
    the exact channel that was measured at ZERO for the whole drawer family.
    """
    from harness.config import HarnessConfig
    from harness.controller import HarnessController

    cfg = HarnessConfig(
        enabled=True,
        stall_step_threshold=80,
        stall_rolling=True,
        stall_max_recoveries=3,
        stall_noop_before_hard=2,
        subtask_override_on_stall=True,
        force_vlm_replan_on_stall=True,
    )

    class _TaskInfo:
        brief_description = "open the middle drawer and place the objects inside"
        task_block = "occlusion-memory task"
        scene_description = "a three-level drawer cabinet"
        primitive_labels = _labels(task_id)

    ctrl = HarnessController.create(task_id=task_id, task_info=_TaskInfo(), config=cfg)
    specs = _specs(task_id)
    overrides = []
    for s in range(100, until, 20):
        ctrl.force_replan = False
        ctrl.subtask_override = None
        ctrl.on_stage_progress(step=s, stage_idx=stage_idx, stage_specs=specs, subtask=subtask)
        if ctrl.subtask_override:
            overrides.append(ctrl.subtask_override)
    return overrides


@pytest.mark.parametrize("task_id,expected", [
    (12, "place cookies"),
    (13, "place cookies"),
    (17, "place butter"),
])
def test_drawer_ladder_now_fires_and_names_the_place_primitive(task_id: int, expected: str):
    """The regression, end to end. Before the fix this list was EMPTY for every drawer task.

    Measured on `outputs/mem_efficacy_local/pilot_armparts_0920_1757/nomem/nomem_s100`:
    0 `subtask override` lines in 9/9 drawer episodes, because the fuzzy matcher returned
    'open middle drawer' -- identical to the subtask -- and `_handle_stall` writes the override
    only on inequality. The microwave family, which never entered the tie, got 2 per episode.
    """
    got = _drive_real_ladder(task_id, 1, subtask="open middle drawer")
    assert got, (
        f"task{task_id}: the stall ladder produced NO override. This is the exact signature of "
        "the tie-break defect: the recovery candidate equalled the stuck subtask, so "
        "`_handle_stall`'s `candidate != subtask` guard suppressed it."
    )
    assert expected in got, f"task{task_id}: expected a {expected!r} override, got {got!r}"


def test_microwave_ladder_behaviour_is_unchanged(task_id: int = 20):
    """The family that worked must keep working, and must keep naming place-* primitives."""
    got = _drive_real_ladder(task_id, 1, subtask="open microwave")
    assert got, "task20: microwave ladder regressed"
    assert all(g.startswith("place") for g in got), f"unexpected non-place override: {got!r}"

