"""The strategy space: the thing all three architectures are missing.

The load-bearing observation, and the reason this module exists:

    VLA success probability is a function of the ATTEMPT, not of the obligation and
    not of the sentence.

GPM, AOM and BOLT each bind at most one textual template to an obligation and vary
only the wording (`serve_ladder`, `admissible_templates`, `templates_for`).  The
archive prices that: BOLT v7 rotated six *paraphrases* of one strategy and scored 0.0
on three seeds; BOLT v9 added the physical variable (does the verb demand a lift?) and
scored 40.0.  A paraphrase is not an attempt.

So a strategy is `(verb_form, approach, hold_steps)` — a coarse physical cell — and the
posterior is P(credited | obligation-family, cell).  Coarseness is the point: with 10
trials per task, only a coarse cell can be estimated at all.

Cross-task strength comes from a hierarchical prior:

    theta_task  ~ N(theta_family, sigma)     <- learned from ~110 archived episodes
    theta_family ~ N(theta_global, sigma)

and selection uses Thompson sampling, which is the *principled* version of the thing
GPM does by accident.  GPM repeats and therefore stumbles into exploration; Thompson
explores on purpose, and stops once a cell is known good — which is what the
gate-heavy arms could never do, because their gates removed the alternative attempts
before they could be tried.

Reference implementations followed for the statistics (not vendored): the Beta-Bernoulli
conjugacy and stratified Thompson sampling of Chapelle & Li (2011), and the
hierarchical shrinkage of Gelman et al., BDA3 ch.5 — partial pooling over families is
exactly their "exchangeable groups" case.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .types import Strategy, classify_approach, classify_verb, RewardRecord

# ---------------------------------------------------------------------------------------
# Attempt vocabulary.  Ordered weakest-assumption first: the first entry of each family is
# the plain attempt, and later entries change a PHYSICAL variable, not a word.
# ---------------------------------------------------------------------------------------
STRATEGY_SPECS: dict[str, list[tuple[str, str, int, str]]] = {
    # family -> [(verb_form, approach, hold_steps, text)]
    #
    # INVARIANT (asserted by selftest A12): for every entry,
    #     classify_verb(text.format(obj=...))    == verb_form
    #     classify_approach(text.format(obj=...)) == approach
    # and the (verb_form, approach, hold_steps) keys are unique inside a family.
    #
    # This is not cosmetic.  `warm()` reads the archive through `classify_verb`/`_approach`
    # on the recorded TEXT, while live selection indexes the declared tuple.  If the two
    # disagree the warmed prior lands in a cell that is never selected, and the whole
    # cross-task transfer silently does nothing.  It also caught a second problem: three
    # pour templates that differed only by the adverb ("pour", "pour slowly", "pour
    # steadily") all collapsed into one cell, which is the paraphrase-versus-attempt
    # mistake in miniature.  The templates below differ in the physical variable.
    "lift": [
        ("pick_up", "plain", 0, "Pick up the {obj}."),
        ("grasp_lift", "plain", 0, "Grasp and lift the {obj}."),
        ("lift", "plain", 0, "Lift the {obj} clear of the source container."),
        ("grasp_lift", "top", 0, "Grasp the {obj} from above, then lift it clear."),
        ("grasp_lift", "top", 8, "Grasp and lift the {obj} from above, then hold it for a moment."),
        ("grasp_lift", "side", 0, "Reach in from the side, grasp the {obj}, and lift it clear."),
    ],
    "pour": [
        ("pour", "plain", 0, "Pour from the {obj} into the target container."),
        ("pour", "tilt", 0, "Tilt the {obj} past horizontal over the target and hold."),
        ("pour", "tilt", 8, "Tilt the {obj} over the target, hold the angle, then tip it back upright."),
    ],
    "place": [
        ("place", "plain", 0, "Place the {obj} into the target container."),
        ("place", "top", 0, "Lower the {obj} from above and release it inside the target container."),
    ],
    "open": [
        ("open", "plain", 0, "Open the next container in order."),
        ("open", "side", 0, "Approach from the side, grip the handle, and slide the container open."),
    ],
    "close": [
        ("close", "plain", 0, "Shut the container that is currently ajar."),
        ("close", "side", 0, "Approach from the side, grip the handle, and slide the container shut."),
    ],
}


def _default_specs(family: str) -> list[tuple[str, str, int, str]]:
    """Two entries, never one.

    A single-candidate family would make "the next attempt must differ" unsatisfiable and
    `_pick` would return None forever, which is the absorbing state this architecture
    exists to remove.  Every family therefore offers at least two physically distinct
    attempts, and selftest asserts it for all families including unknown ones.
    """
    return [
        ("other", "plain", 0, "Retry the current step on the {obj}, aiming at the same spot."),
        ("other", "top", 0, "Retry the current step on the {obj}, approaching from above."),
    ]


@dataclass
class _Arm:
    """Beta posterior for one (family, strategy-cell)."""
    a: float = 1.0
    b: float = 1.0
    n: int = 0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.a, self.b)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)


@dataclass
class StrategyBank:
    """Posteriors over strategy cells, pooled across tasks by family.

    `warm` folds in the archived reward stream, which is the only reason a 10-trial
    budget can carry a learned prior at all.  Records whose obligation family is
    unknown still contribute through their own family key, so no data is wasted.
    """
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    # family -> cell-key -> arm
    cells: dict[str, dict[tuple[str, str, int], _Arm]] = field(default_factory=dict)
    # family-level pooled prior (the shrinkage target)
    family_prior: dict[str, float] = field(default_factory=dict)
    global_prior: float = 0.35
    n_records: int = 0
    n_credited: int = 0
    shrinkage: float = 8.0          # pseudo-counts pulling a cell toward its family mean
    history: list[tuple[str, str, bool]] = field(default_factory=list)

    # -- prior construction ----------------------------------------------------------------
    def warm(self, records: Iterable[RewardRecord]) -> int:
        """Fit the hierarchical prior from attributed archive records."""
        by_fam: dict[str, list[int]] = {}
        by_cell: dict[str, dict[tuple[str, str, int], list[int]]] = {}
        for r in records:
            if r.outcome not in ("credited", "stalled", "active"):
                continue
            fam = r.family or "other"
            credit = 1 if r.outcome == "credited" else 0
            by_fam.setdefault(fam, []).append(credit)
            by_cell.setdefault(fam, {}).setdefault(r.strategy_key, []).append(credit)
            self.n_records += 1
            self.n_credited += credit

        allv = [v for vs in by_fam.values() for v in vs]
        self.global_prior = (sum(allv) / len(allv)) if allv else 0.35

        # partial pooling: each family mean is pulled toward the global mean by the
        # number of observations, which is what keeps a 3-sample family from dominating.
        k = 6.0
        for fam, vs in by_fam.items():
            n, s = len(vs), sum(vs)
            self.family_prior[fam] = (s + k * self.global_prior) / (n + k)

        for fam, cells in by_cell.items():
            base = self.family_prior.get(fam, self.global_prior)
            for cell, vs in cells.items():
                n, s = len(vs), sum(vs)
                # beta-binomial with the family mean as prior -> shrunk cell estimate
                arm = _Arm(a=1.0 + s, b=1.0 + (n - s))
                arm.n = n
                # blend toward the family prior for thin cells via extra pseudo-counts
                g = max(0.0, self.shrinkage - n) / max(1.0, self.shrinkage)
                if g > 0:
                    arm.a += g * self.shrinkage * base
                    arm.b += g * self.shrinkage * (1.0 - base)
                self.cells.setdefault(fam, {})[cell] = arm
        return self.n_records

    # -- read -----------------------------------------------------------------------------
    def candidates(self, family: str) -> list[tuple[str, str, int, str]]:
        return STRATEGY_SPECS.get(family) or _default_specs(family)

    def arm(self, family: str, key: tuple[str, str, int]) -> _Arm:
        fam = self.cells.setdefault(family, {})
        if key not in fam:
            base = self.family_prior.get(family, self.global_prior)
            pseudo = 3.0
            fam[key] = _Arm(a=1.0 + pseudo * base, b=1.0 + pseudo * (1.0 - base))
        return fam[key]

    # -- selection ------------------------------------------------------------------------
    def select(self, family: str, obj: str, *, attempt: int = 0,
               exclude: set[tuple[str, str, int]] | None = None) -> Strategy:
        """Thompson-sample the next attempt for `family`.

        `exclude` is how the scheduler forbids repeating the failed attempt: unlike a
        gate, it *removes nothing from the environment*, it only changes which attempt
        goes next.  This is the structural difference from GPM/AOM/BOLT, whose gates
        rewrite onto the same obligation and can therefore absorb.
        """
        cands = self.candidates(family)
        excl = set(exclude or ())
        pool = [c for c in cands if (c[0], c[1], c[2]) not in excl] or cands
        # deterministic first attempt per obligation keeps behaviour auditable
        if attempt <= 0:
            v, ap, h, tmpl = pool[0]
            return Strategy(obligation=family, verb_form=v, approach=ap,
                            hold_steps=h, text=tmpl.format(obj=obj))
        best, best_s = None, -1.0
        for (v, ap, h, tmpl) in pool:
            s = self.arm(family, (v, ap, h)).sample(self.rng)
            if s > best_s:
                best, best_s = (v, ap, h, tmpl), s
        v, ap, h, tmpl = best
        return Strategy(obligation=family, verb_form=v, approach=ap,
                        hold_steps=h, text=tmpl.format(obj=obj))

    def observe(self, family: str, key: tuple[str, str, int], credited: bool) -> None:
        arm = self.arm(family, key)
        if credited:
            arm.a += 1.0
        else:
            arm.b += 1.0
        arm.n += 1
        self.history.append((family, "/".join(str(x) for x in key), credited))

    def ranking(self, family: str) -> list[tuple[tuple[str, str, int], float, int]]:
        out = [(k, a.mean, a.n) for k, a in self.cells.get(family, {}).items()]
        out.sort(key=lambda x: -x[1])
        return out

    def snapshot(self) -> dict:
        return {
            "global_prior": round(self.global_prior, 4),
            "family_prior": {k: round(v, 4) for k, v in self.family_prior.items()},
            "n_records": self.n_records,
            "n_credited": self.n_credited,
            "n_cells": sum(len(v) for v in self.cells.values()),
            "top": {f: [(list(k), round(m, 3), n) for k, m, n in self.ranking(f)[:3]]
                    for f in list(self.cells)[:6]},
        }


def strategy_of(text: str, obligation: str) -> Strategy:
    """Classify an arbitrary instruction into the same cell space (for offline scoring)."""
    return Strategy(obligation=obligation, verb_form=classify_verb(text),
                    approach=classify_approach(text), hold_steps=0, text=str(text))
