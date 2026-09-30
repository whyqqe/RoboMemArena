"""The bottleneck posterior.

One object answers one question: *why is credit not moving?*  Every downstream
behaviour — whether the planner resamples a strategy, whether memory is queried,
whether the harness does nothing at all — is a function of this distribution.

Why a posterior rather than a rule: the three architectures all encode "why" as a
boolean threshold (`attempts >= K`, `stall >= K`, `stall_limit`) and then act as if it
were certain.  BOLT's own history prices that mistake: its guard collapsed when 15 of
17 steps were one repeated sentence, because a *count* cannot tell a dead instruction
from a slow scorer from an unsighted object.  A posterior can, because those three
produce different likelihoods on observables that the harness already records:

    enacted but not credited, and age < lambda_hat      -> UNSYNCED
    executed, zero displacement, retain success          -> UNGROUNDED
    retract / slip / collision on the action channel      -> UNSAFE
    no observation of the referent at all                -> UNKNOWN
    otherwise, repeated attempts with no credit           -> UNREACHABLE

`lambda_hat` (the expected enacted->credited lag) is *estimated*, not assumed, from
whatever credited records carry a measured lag.  That single quantity is the whole
difference between waiting and resampling, and no arm in the archive computes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .types import (
    CAUSES,
    CAUSE_UNGROUNDED,
    CAUSE_UNKNOWN,
    CAUSE_UNREACHABLE,
    CAUSE_UNSAFE,
    CAUSE_UNSYNCED,
)

# Likelihoods P(observable | cause).  These are hand-set from the archive's observed
# signatures and are the one place a domain assumption enters; `falsify.py` checks them
# against held-out episodes rather than trusting them.
_LIKELIHOOD: dict[str, dict[str, float]] = {
    # observable            UNREACH  UNKNOWN  UNGROUND  UNSYNC  UNSAFE
    "credited":            {"UNREACHABLE": 0.02, "UNKNOWN": 0.05, "UNGROUNDED": 0.05,
                            "UNSYNCED": 0.60, "UNSAFE": 0.02},
    "enacted_no_credit":   {"UNREACHABLE": 0.45, "UNKNOWN": 0.25, "UNGROUNDED": 0.30,
                            "UNSYNCED": 0.80, "UNSAFE": 0.20},
    # The distinction that makes the filter work.  `enacted_no_credit` means "acted, and the
    # scorer is not due yet" and is weak evidence for UNSYNCED.  Once the age passes
    # lambda_hat, the same silence means the opposite: the scorer WAS due and did not
    # credit the attempt, which is evidence that the attempt itself failed.  Feeding
    # `enacted_no_credit` forever pins the posterior on UNSYNCED (likelihood 0.80 against
    # ~0.3 for everything else, applied every step), the scheduler never leaves RESYNC or
    # PERSIST, and exploration stops — an absorbing state introduced by the estimator
    # rather than by the action space.  Measured by selftest A5/C1 before this split existed.
    "overdue_no_credit":   {"UNREACHABLE": 0.82, "UNKNOWN": 0.22, "UNGROUNDED": 0.38,
                            "UNSYNCED": 0.02, "UNSAFE": 0.30},
    "zero_displacement":   {"UNREACHABLE": 0.20, "UNKNOWN": 0.10, "UNGROUNDED": 0.75,
                            "UNSYNCED": 0.10, "UNSAFE": 0.45},
    "retract_or_slip":     {"UNREACHABLE": 0.10, "UNKNOWN": 0.02, "UNGROUNDED": 0.25,
                            "UNSYNCED": 0.02, "UNSAFE": 0.85},
    "no_observation":      {"UNREACHABLE": 0.20, "UNKNOWN": 0.80, "UNGROUNDED": 0.35,
                            "UNSYNCED": 0.05, "UNSAFE": 0.05},
    "repeat_no_move":      {"UNREACHABLE": 0.80, "UNKNOWN": 0.15, "UNGROUNDED": 0.30,
                            "UNSYNCED": 0.10, "UNSAFE": 0.20},
    "lag_within_lambda":   {"UNREACHABLE": 0.02, "UNKNOWN": 0.02, "UNGROUNDED": 0.02,
                            "UNSYNCED": 0.95, "UNSAFE": 0.02},
    "memory_hit":          {"UNREACHABLE": 0.20, "UNKNOWN": 0.02, "UNGROUNDED": 0.30,
                            "UNSYNCED": 0.10, "UNSAFE": 0.10},
}
_UNIFORM = 1.0 / len(CAUSES)


@dataclass
class BottleneckBelief:
    """beta_t — posterior over why the current obligation is not moving.

    Deliberately a *filter*, not a fitter: it must behave at step 5 of a 10-trial
    budget, before any within-task statistics exist.  Cross-task strength comes from
    `strategy.py`'s hierarchical prior, not from here.
    """
    prior: dict[str, float] = field(default_factory=lambda: {c: _UNIFORM for c in CAUSES})
    beta: dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    # Archive-derived defaults, NOT guesses.  `attribution.py` measures, over the credited
    # attempts in the archived runs, an enacted->credited lag with median 25, mean 53.6,
    # p90 145 and max 685 steps.  The distribution is heavy-tailed, so the median is the
    # right central estimate and p90 is the right patience bound.  (The measured quantity
    # is the gap between successive attempts inside a credited stage's window, which is the
    # available proxy for the scorer's observation cadence.)
    lambda_hat: float = 25.0
    lambda_p90: float = 60.0
    # `escape_horizon = escape_multiple * hold_horizon`.  2.5 is chosen against the measured
    # tail, not picked for roundness: hold_horizon starts at 60 (nearest-rank p90 of the
    # archive lag distribution) and rises to ~130 once lags are observed, so the escape
    # fires at 150 and then ~325 steps.  The wasteful direction is escaping EARLY (that
    # cancels an attempt that was about to be credited: T3 puts that at up to 79.8% of
    # successful attempts under the other arms' 8- and 12-step horizons), so this is
    # deliberately the patient side of the trade.
    escape_multiple: float = 2.5
    # Forgetting factor, and it is load-bearing rather than decorative.  A pure
    # multiplicative filter drives the losing causes to 0 in float, after which no amount
    # of contradictory evidence can bring them back: 30 observations of one lag type push
    # P(UNSYNCED) to 1.0, and a later likelihood of 0.02 multiplied into 1.0 still yields
    # 1.0 after normalisation.  That is an absorbing state inside the estimator — exactly
    # the failure this architecture exists to remove — so each update leaks a little mass
    # back toward the prior and the filter stays recoverable.  Measured by selftest A4.
    leak: float = 0.05
    lags: list[float] = field(default_factory=list)
    _MAX_LAGS: int = 256
    history: list[tuple[str, dict[str, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.beta:
            self.beta = dict(self.prior)

    # -- estimation ------------------------------------------------------------------------
    def observe_lag(self, lag_steps: int) -> None:
        """Feed a measured enacted->credited lag.  This is the UNSYNCED timescale.

        Median for the central estimate and p90 for the patience bound, because the
        measured distribution is heavy-tailed (p90 = 145 against a median of 25).  An
        exponential-mean update would let one 685-step episode inflate the estimate and
        make the arm hold for the rest of the run.
        """
        if lag_steps is None or lag_steps < 0:
            return
        self.lags.append(float(lag_steps))
        if len(self.lags) > self._MAX_LAGS:
            self.lags = self.lags[-self._MAX_LAGS:]
        s = sorted(self.lags)
        self.lambda_hat = max(1.0, s[len(s) // 2])
        # nearest-rank p90 (not the interpolated percentile): this is a PATIENCE bound, and
        # a bound that sits below an observed sample would cancel that sample's attempt.
        idx = min(len(s) - 1, max(0, math.ceil(0.9 * len(s)) - 1))
        self.lambda_p90 = max(self.lambda_hat, s[idx])

    @property
    def hold_horizon(self) -> float:
        """Age up to which an attempt may plausibly still be in flight.

        This is the quantity the RESYNC veto and the `enacted_no_credit` observable must use,
        and it is deliberately a HIGH quantile, not the median.  If the hold window were
        `lambda_hat` (the median lag), then by construction half of all attempts would be
        flagged overdue before their credit arrived, and the scheduler would cancel work
        that was about to be credited — the exact T3 failure.  With the archive's measured
        spread (median 25, p90 145) that mistake would cancel most successful attempts.
        """
        return max(self.lambda_p90, self.lambda_hat * 1.25, 1.0)

    @property
    def escape_horizon(self) -> float:
        """Age beyond which an attempt with no credit is declared dead.

        Strictly above `hold_horizon` so that there is never a state where the scheduler
        both refuses to act and refuses to escape.
        """
        return max(self.escape_multiple * self.hold_horizon, self.hold_horizon + 1.0)

    # -- filtering -------------------------------------------------------------------------
    def update(self, observables: dict[str, bool], *, strength: float = 1.0) -> dict[str, float]:
        """Bayesian update from *coarse boolean observables* (see `policy.py` for who computes them)."""
        post = dict(self.beta)
        for obs, present in observables.items():
            if not present:
                continue
            table = _LIKELIHOOD.get(obs)
            if not table:
                continue
            for c in CAUSES:
                post[c] = post[c] * (table[c] ** strength)
        total = sum(post.values()) or 1.0
        post = {c: post[c] / total for c in CAUSES}
        # leak toward the prior so the filter can never lock out a cause (see `leak`)
        if self.leak > 0:
            post = {c: (1.0 - self.leak) * post[c] + self.leak * self.prior[c]
                    for c in CAUSES}
            total = sum(post.values()) or 1.0
            post = {c: post[c] / total for c in CAUSES}
        self.beta = post
        self.n_observations += 1
        self.history.append((max(post, key=post.get), dict(post)))
        if len(self.history) > 64:
            self.history = self.history[-64:]
        return post

    def reset(self, *, keep_lambda: bool = True) -> None:
        """New obligation: the evidence was about the old one.  The lag estimate is
        task-level and survives, because the scorer's cadence does not change per stage."""
        self.beta = dict(self.prior)
        self.n_observations = 0
        self.history.clear()
        if not keep_lambda:
            self.lambda_hat = 25.0
            self.lambda_p90 = 60.0
            self.lags = []

    # -- reads -----------------------------------------------------------------------------
    def argmax(self) -> str:
        return max(self.beta, key=self.beta.get)

    def p(self, cause: str) -> float:
        return float(self.beta.get(cause, 0.0))

    def entropy(self) -> float:
        import math
        return -sum(p * math.log(p + 1e-12) for p in self.beta.values())

    def is_unsynced(self, *, enacted_age_steps: int) -> bool:
        """Waiting is correct exactly here: acted, scorer not due yet."""
        return (self.p(CAUSE_UNSYNCED) >= 0.35
                and 0 <= enacted_age_steps <= self.lambda_hat)

    def snapshot(self) -> dict[str, Any]:
        return {
            "argmax": self.argmax(),
            "beta": {k: round(v, 4) for k, v in self.beta.items()},
            "entropy": round(self.entropy(), 4),
            "lambda_hat": round(self.lambda_hat, 2),
            "lambda_p90": round(self.lambda_p90, 2),
            "escape_horizon": round(self.escape_horizon, 2),
            "n_lag_samples": len(self.lags),
            "n_observations": self.n_observations,
        }


def observables(*, credited: bool = False, enacted_no_credit: bool = False,
                overdue_no_credit: bool = False, zero_displacement: bool = False,
                retract_or_slip: bool = False, no_observation: bool = False,
                repeat_no_move: bool = False, lag_within_lambda: bool = False,
                memory_hit: bool = False) -> dict[str, bool]:
    """Named constructor so callers cannot typo an observable into silence."""
    return {
        "credited": credited,
        "enacted_no_credit": enacted_no_credit,
        "overdue_no_credit": overdue_no_credit,
        "zero_displacement": zero_displacement,
        "retract_or_slip": retract_or_slip,
        "no_observation": no_observation,
        "repeat_no_move": repeat_no_move,
        "lag_within_lambda": lag_within_lambda,
        "memory_hit": memory_hit,
    }
