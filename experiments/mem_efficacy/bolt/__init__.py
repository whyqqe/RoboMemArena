"""BOLT-Sync: Belief–Obligation–Ledger–Telemetry with Clock Synchronization.

Package layout (Phase-1 core + closable Phase-2/3 stubs):

  types.py     enums, ActionContract, DecisionNeed, VerifyResult
  claim.py     Claim + BeliefState (invalidation-aware)
  clock.py     dual-clock SyncState (cognition/action/evidence ages, commitment gap)
  graph.py     ObligationGraph seeded from BDDL `primitive_order`
  ledger.py    VerifiedLedger (attempt / verified / ambiguous / failed)
  need.py      DecisionNeedCompiler
  router.py    rule-based EvidenceRouter (U(m) stub)
  arbiter.py   UnifiedArbiter (six-layer check)
  verify.py    IndependentVerifier
  board.py     prompt board renderer
  harness.py   BoltHarness — the S_t controller
  bind.py      planner hooks (installed via memexp_bolt_bind / sitecustomize)
  telemetry.py shared counters

Reference: repo-root BOLT.md. Reuses GPM templates via memexp_evmem and AOM's
execution-partition lessons (soft grasp→place deps, scoring bridge).
"""
from __future__ import annotations

from .harness import BoltHarness, enabled
from .types import ActionContract, DecisionNeed, VerifyResult

__all__ = [
    "BoltHarness",
    "ActionContract",
    "DecisionNeed",
    "VerifyResult",
    "enabled",
]
