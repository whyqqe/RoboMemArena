"""DIAL — Diagnosis-Informed Action Law.

One bottleneck posterior drives Planner, VLA strategy, Proactive Memory and Harness
together.  The design argument is in one sentence:

    GPM, AOM and BOLT each schedule *inside* one component.  None of them schedules
    ACROSS components, so each guesses "what should happen now" from its own stale
    evidence.  Archive evidence says this is what costs the score:

        arm     t5    t8    t22   t19   mean
        GPM    20.0  46.7  73.3  26.7  41.7     <- schedule: counters
        AOM    45.0  50.0  33.3  23.3  37.9     <- schedule: one law, inside obligations
        BOLT   13.8  40.0  40.0  23.3  29.3     <- schedule: three subsystems

    Wider legality gates scored strictly worse (GPM 4 rules > AOM dep-check > BOLT
    dep+ordinal+satisfiability+verbatim), and BOLT's own history shows both failure
    classes are worth ~30 points: v1->v5 +30 (dependency semantics) and v7->v9 +27
    (interface/leak).  A gate that rewrites a proposal onto the SAME obligation is an
    absorbing state whenever that obligation is not physically executable — measured
    as BOLT v8 seeds 100/102/104 and AOM t22 seeds 101/102/106/107/108, all 0.0 with
    the same "Lift" family never credited.

DIAL's structural answer: make rejection impossible.  The action space has no BLOCK,
only RESAMPLE (a genuinely different strategy for the same obligation), QUERY (the
memory channel), RESYNC (do nothing; the scorer has not caught up) and PERSIST.

Layers:
    types.py        Strategy / RewardRecord / Cause / BottleneckBelief
    attribution.py  archived-run -> RewardRecord stream (the harness already logs it)
    bottleneck.py   observable evidence -> posterior over why credit is not moving
    strategy.py     the per-obligation strategy space + Thompson sampling
    policy.py       the single scheduling law
    bind.py         harness + planner integration (the only stateful seam)
"""
from __future__ import annotations

from .types import (  # noqa: F401
    CAUSE_UNGROUNDED,
    CAUSE_UNKNOWN,
    CAUSE_UNREACHABLE,
    CAUSE_UNSAFE,
    CAUSE_UNSYNCED,
    CAUSES,
    RewardRecord,
    Strategy,
)
from .bottleneck import BottleneckBelief  # noqa: F401
from .policy import DIALPolicy, DIALDecision  # noqa: F401
from .strategy import StrategyBank  # noqa: F401

__all__ = [
    "Strategy",
    "RewardRecord",
    "BottleneckBelief",
    "StrategyBank",
    "DIALPolicy",
    "DIALDecision",
    "CAUSES",
    "CAUSE_UNREACHABLE",
    "CAUSE_UNKNOWN",
    "CAUSE_UNGROUNDED",
    "CAUSE_UNSYNCED",
    "CAUSE_UNSAFE",
]
