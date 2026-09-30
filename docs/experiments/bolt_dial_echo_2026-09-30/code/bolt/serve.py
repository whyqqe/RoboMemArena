"""Serve policy: what the robot is actually told to do, as a pure function.

BOLT's architecture has three layers and they must not be mixed:

  invariants   (graph.py)   — what may gate what; closure / ordinality / reachability
  serve policy (this file)  — the *words* put in front of the VLA for one obligation
  discipline   (arbiter.py) — refuse a proposal that breaks obligation order

v1 collapsed the serve policy into ad-hoc strings scattered across `graph.py`
and `arbiter.py`.  The measured cost was severe:

  * the ladder SATURATED.  `serve_text` clamped its index at `len(ladder)-1`, so
    after 5 directions the same sentence came back forever.  On the archived v5
    run of task 8 this produced 14/17 and 17/19 identical `pick tomato sauce.`
    outputs in the trials that failed to lift — while the control arm (GPM), which
    phrases the same intent differently every step, lifted in the same scenes and
    scored 46.7 against BOLT's 30.0.  Per-step variation is not cosmetic for a VLA:
    an identical repeated command reproduces an identical failed trajectory.
  * the BDDL label (`pick tomato sauce.`) was itself a rung.  It is a graph
    identifier, not a robot command, and it reads worse than any natural phrasing.
  * pouring nodes had no ordinal awareness, so nothing kept "pour a second time"
    distinct from "pour".

So: phrases live here, one function per phase, ordered weakest-assumption first.
`ObligationGraph.serve_text` cycles through them and never returns the same
sentence twice in a row.  Nothing here can gate the graph; it only changes words.
"""
from __future__ import annotations

import re

# Phases whose objects are held in a vessel worth naming ("tomato sauce bottle").
# Naming the container is what the control arm does and it measurably helps
# grounding: "grasp the tomato sauce" is ambiguous between the bottle, the jar on
# the shelf and the sauce already in the pan.
_VESSEL_WORDS = ("sauce", "milk", "juice", "water", "oil", "wine", "syrup", "soup")


def _place(target: str, fallback: str = "target container") -> str:
    """Bare noun phrase for a destination, without a leading article.

    Callers already write "the {t}", so an article must not be carried through twice
    (the seeded target is literally "the target container", which produced
    "pour over the the target container" in the served command).
    """
    t = re.sub(r"^(the|a|an)\s+", "", str(target or "").strip(), flags=re.I)
    return t.strip() or fallback


def vessel_name(obj: str, *, is_pour_substance: bool) -> str:
    """Natural noun phrase for the object, naming the vessel when it is poured."""
    o = str(obj or "").strip()
    if not o:
        return "the target object"
    if not is_pour_substance:
        return o
    low = o.lower()
    if "bottle" in low or "container" in low:
        return o
    if any(w in low for w in _VESSEL_WORDS):
        return f"{o} bottle"
    return o


def command_ladder(phase: str, obj: str, target: str, *, is_pour_substance: bool = False,
                   pour_ordinal: int = 0) -> list[str]:
    """Ordered, diverse, risk-aware commands for one obligation.

    Ordered so that the FIRST entry is the most standard attempt and later entries
    progressively change strategy (re-localize, re-approach, re-grip).  All entries
    are complete imperatives: no graph identifiers, no symbolic predicates.
    """
    v = vessel_name(obj, is_pour_substance=is_pour_substance)
    t = _place(target)
    ph = str(phase or "")

    if ph == "grasp":
        # INVARIANT I5 — every rung must be able to satisfy the stage the node is
        # verified by.  A `grasp` obligation is verified by the environment's Lift
        # stage (`01_Lift_*`), so every rung MUST ask for a lift.
        #
        # Measured regression (v7, t8): three of six rungs said only "reach for and
        # grasp", "re-localize … close the gripper", "approach … grip firmly".  The
        # VLA complied exactly — it grasped and did NOT lift — so the Lift stage was
        # never credited.  The Arbiter's obligation discipline then rewrote every
        # attempt back onto that same node, and all three trials ran ~2400 steps and
        # 18-19 identical failures for 0.0.  The control arm never makes this mistake:
        # every one of its rungs reads "grasp … and lift it", which is why it lifts in
        # the same scenes.  v5 scored 6/10 lifts for the same reason — its saturating
        # rung was `pick tomato sauce`, and `pick` implies the lift.
        return [
            f"Pick up the {v}.",
            f"Grasp and lift the {v}.",
            f"Reach for and grasp the {v}, then lift it clear of the source container.",
            f"Lift the {v} clear of the source container.",
            f"Approach the {v} from above, close the gripper, and raise it.",
            f"Grasp the {v} firmly and lift it until it is raised off the table.",
        ]
    if ph == "place":
        return [
            f"Place the {v} into the {t}.",
            f"Move the {v} over the {t}, then lower and release it.",
            f"Regrasp the {v} and place it into the {t}.",
            f"Release the {v} only after it is inside the {t}.",
        ]
    if ph.startswith("pour"):
        if pour_ordinal >= 2:
            return [
                f"Tilt the {v} over the {t} and pour a second time.",
                f"Pour the contents of the {v} over the {t} again.",
                f"Regrasp the {v} upright, then tilt it back over the {t} "
                f"and pour once more.",
                f"Pour slowly from the {v} over the {t} a second time.",
            ]
        return [
            f"Tilt the {v} over the {t} and pour.",
            f"Pour the contents of the {v} over the {t}.",
            f"Regrasp the {v} upright, then tilt it over the {t} to pour.",
            f"Pour slowly from the {v} over the {t}.",
        ]
    if ph == "open":
        return [
            f"Pull the {t} open.",
            f"Re-approach the handle of the {t} and open it.",
            f"Grip the handle and slide the {t} open.",
        ]
    if ph == "close":
        return [
            f"Push the {t} closed.",
            f"Re-approach the handle of the {t} and close it.",
            f"Grip the handle and slide the {t} shut.",
        ]
    return [f"Retry the current step on the {v}."]


def is_pour_phase(phase: str) -> bool:
    return str(phase or "").startswith("pour")


# Verbs that can actually satisfy the stage a phase is verified by.  Note that
# "grasp" is deliberately NOT a lift verb: the environment credits a Lift stage only
# when the object leaves the table, and a grasp-only command leaves it exactly there.
_SATISFYING = {
    "grasp": re.compile(r"\b(pick|lift|raise|hoist|clear)\b", re.I),
    "place": re.compile(r"\b(place|put|insert|release|inside|into|drop)\b", re.I),
    "pour": re.compile(r"\b(pour|tilt|tip|decant)\b", re.I),
    "open": re.compile(r"\bopen\w*\b", re.I),
    "close": re.compile(r"\b(clos\w*|shut\w*)\b", re.I),
}


def satisfies_stage(phase: str, text: str) -> bool:
    """True when `text` asks for the action the phase's scored stage rewards.

    INVARIANT I5.  A served command that cannot satisfy its own obligation is not a
    weak attempt, it is a guaranteed dead step: the environment has nothing to credit,
    so Φ never drops, the Arbiter keeps the same node ACTIVE, and the episode is spent
    re-issuing an instruction that cannot succeed.  This is checked rather than trusted
    because the failure is invisible in any per-step metric — the planner looks
    responsive, the command looks reasonable, and only the score is 0.
    """
    ph = str(phase or "")
    for key, rx in _SATISFYING.items():
        if ph.startswith(key):
            return bool(rx.search(str(text or "")))
    return True


def violating_rungs(phase: str, ladder: list[str]) -> list[str]:
    """Rungs that cannot satisfy `phase`'s scored stage (empty when all are fine)."""
    return [t for t in ladder if not satisfies_stage(phase, t)]


def pour_ordinal(phase: str) -> int:
    ph = str(phase or "")
    if ph == "pour_two":
        return 2
    if ph == "pour_one":
        return 1
    return 0
