"""DIAL types — the small vocabulary everything else is expressed in.

There is one design rule here, inherited from the three architectures' measured
failures: a *strategy* is a physical attempt, not a paraphrase.  GPM, AOM and BOLT
all varied the sentence and called it variation; the archive says the phrase is not
the variable that matters, the attempt is.  So `Strategy` carries approach geometry,
verb strength and hold time, and the text is only the surface form of those.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------------------
# Bottleneck causes.  These are the five reasons credit can fail to move, and they are
# *distinguishable from the outside* — that is the whole point.  A cause that could only
# be known by asking the model would not belong here.
# ---------------------------------------------------------------------------------------
CAUSE_UNREACHABLE = "UNREACHABLE"    # the VLA cannot do it as phrased/approached
CAUSE_UNKNOWN = "UNKNOWN"            # no observation of the target yet (memory channel)
CAUSE_UNGROUNDED = "UNGROUNDED"      # acting, but on the wrong referent / no displacement
CAUSE_UNSYNCED = "UNSYNCED"          # enacted, but the scorer has not caught up
CAUSE_UNSAFE = "UNSAFE"              # physical failure: slip, collision, retract

CAUSES = (
    CAUSE_UNREACHABLE,
    CAUSE_UNKNOWN,
    CAUSE_UNGROUNDED,
    CAUSE_UNSYNCED,
    CAUSE_UNSAFE,
)

# Attractors: the coarse physical class of an attempt.  Reusing `attractor_cluster`'s
# insight (memexp_evmem) but keyed on the attempt rather than the sentence.
_APPROACH_RX = {
    "top": re.compile(r"from above|overhead|descend|from the top", re.I),
    "side": re.compile(r"from the side|sideways|lateral|from the left|from the right", re.I),
    "tilt": re.compile(r"tilt|tip|angle|incline", re.I),
}
_VERB_RX = {
    "pick_up": re.compile(r"\bpick up\b", re.I),
    "grasp_lift": re.compile(r"\bgrasp\b.*\b(lift|raise)\b", re.I),
    "lift": re.compile(r"\b(lift|raise|hoist)\b", re.I),
    "grasp": re.compile(r"\b(grasp|grip|secure)\b", re.I),
    "reach": re.compile(r"\breach\b", re.I),
    "pour": re.compile(r"\b(pour|tilt|decant)\b", re.I),
    "place": re.compile(r"\b(place|put|insert|release)\b", re.I),
    "open": re.compile(r"\bopen\w*\b", re.I),
    "close": re.compile(r"\b(clos\w*|shut\w*)\b", re.I),
}
_LIFT_RX = re.compile(r"\b(pick up|lift|raise|hoist)\b", re.I)


@dataclass(frozen=True)
class Strategy:
    """One physical attempt at an obligation.

    `sid` is the key the posterior is indexed by, and it is deliberately coarse: the
    archive has ~110 episodes, so a key fine enough to distinguish sentences would
    leave every cell with a handful of samples.  Coarse keys are what make the
    hierarchical prior in `strategy.py` able to learn anything at 10 trials/task.
    """
    obligation: str
    verb_form: str          # pick_up | grasp_lift | lift | grasp | reach | pour | ...
    approach: str           # top | side | tilt | plain
    hold_steps: int         # how long to hold after the manipulation before releasing
    text: str               # surface form handed to the VLA

    @property
    def sid(self) -> str:
        return f"{self.verb_form}/{self.approach}/h{self.hold_steps}"

    def key(self) -> tuple[str, str, int]:
        return (self.verb_form, self.approach, self.hold_steps)


def classify_verb(text: str) -> str:
    """Coarse verb class of an attempt, strongest-first (order matters)."""
    for name in ("pick_up", "grasp_lift", "lift", "pour", "place", "open", "close",
                 "grasp", "reach"):
        if _VERB_RX[name].search(str(text or "")):
            return name
    return "other"


def classify_approach(text: str) -> str:
    for name, rx in _APPROACH_RX.items():
        if rx.search(str(text or "")):
            return name
    return "plain"


def has_lift_word(text: str) -> bool:
    """True when the attempt asks the object to leave the surface.

    This is the single most load-bearing predicate in the archive: task 8/22's first
    scored stage is `01_Lift_*`, so an attempt without a lift word cannot be credited
    no matter how good the grasp is.  Measured: BOLT v8 seeds 100/102/104 emitted
    grasp-only text and scored 0.0 while the control arm lifted in the same scenes.
    """
    return bool(_LIFT_RX.search(str(text or "")))


@dataclass
class RewardRecord:
    """One attributed attempt: the unit the posterior is fit on.

    Produced by `attribution.py` from the harness's own `episode_evidence` log, which
    already records (instruction, outcome, stage) per attempt.  The harness is the only
    component that knows which stage moved after which attempt, so it is the only
    honest reward channel — no arm in the archive uses it as one.
    """
    arm: str
    task: int
    seed: int
    episode: int
    step: int
    obligation: str            # stage the attempt was aimed at (from stall notes)
    text: str                  # what the VLA was told
    outcome: str               # "active" | "stalled" | "credited"
    family: str = ""           # OBLIGATION_FAMILY: lift | pour | place | open | close
    trial_score: float = 0.0   # episode-level score, for context
    credited_delta: float = 0.0
    lag_steps: int = -1        # enacted -> credited delay, -1 when unknown
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_key(self) -> tuple[str, str, int]:
        return (classify_verb(self.text), classify_approach(self.text), 0)

    @property
    def success(self) -> bool:
        return self.outcome == "credited"

    @property
    def stall(self) -> bool:
        return self.outcome == "stalled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "task": self.task, "seed": self.seed, "episode": self.episode,
            "step": self.step, "obligation": self.obligation, "family": self.family,
            "text": self.text, "outcome": self.outcome, "strategy": "/".join(
                str(x) for x in self.strategy_key),
            "verb_form": self.strategy_key[0], "approach": self.strategy_key[1],
            "trial_score": self.trial_score, "lag_steps": self.lag_steps,
        }


_STAGE_RX = re.compile(r"([0-9]{2}_[A-Za-z0-9_]+?)\s*:\s*(\d+)")


def family_of(obligation: str) -> str:
    """Scored-stage name -> obligation family.  Used to pool evidence across tasks."""
    low = str(obligation or "").lower()
    if re.search(r"lift|grasp|pick", low):
        return "lift"
    if re.search(r"pour", low):
        return "pour"
    if re.search(r"place|put", low):
        return "place"
    if "open" in low:
        return "open"
    if "close" in low:
        return "close"
    return "other"


def parse_stage_note(note: str) -> tuple[str, int]:
    """`"01_Lift_Tomato_Sauce:80"` -> ("01_Lift_Tomato_Sauce", 80)."""
    m = _STAGE_RX.search(str(note or ""))
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))
