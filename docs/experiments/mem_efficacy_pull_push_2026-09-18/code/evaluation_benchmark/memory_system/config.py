from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySystemConfig:
    """Practically optimizable memory hyperparameters (no weight training required)."""

    stage_anchor: bool = False
    salience_subtask_change: bool = True
    bank_max: int = 8
    cluster_distance: int = 4
    recent_window: int = 7
    # Approximate number of temporally SPREAD frames to union into the bank, on top of whatever
    # the anchor-based builder produced. 0 disables it, so the official behaviour is unchanged
    # unless an arm declares it.
    #
    # WHY IT EXISTS: `merge_keyframe_bank` draws candidates only from stage boundaries and
    # subtask changes, so its bank is a handful of anchors clustered around TRANSITIONS and never
    # a spread across the episode. Job 593817 measured what that costs -- 1.7 frames injected per
    # call against a `bank_max` of 8, i.e. the bank was CANDIDATE-limited, not cap-limited, and
    # the `cluster_distance`/`bank_max` sweep could not move it. A stride over the frame store is
    # a pure function of `(recent_start, frame_store_main)` and of nothing any model wrote, so it
    # is available to a Planner that nominates no keyframes at all -- which is exactly the API
    # Planner's situation.
    #
    # It is a FALLBACK for a planner that does not nominate, not a replacement for nomination:
    # a stride is information-BLIND (the same frames for task 4 and task 26) whereas a nomination
    # is task-adaptive. See `nomination_prompt`.
    kf_spread: int = 0
    # Teach the Planner WHAT to nominate in `keyframe_positions`. 0 keeps the previous wording.
    #
    # PrediMem's keyframe bank is sourced from the model's OWN nominations, and the paper's
    # ablation (bank removed: TSR 38.5% -> 17.7%) is the reason that channel is load-bearing.
    # With the local PrediMem VLM the nominations come from TRAINING. The API Planner is not
    # trained for it, so it emits `keyframe_positions: []` on every step (measured: J_hist is a
    # list of empty lists, `kf_n = 0`, job 586700) and `build_visual_memory` therefore returns
    # nothing -- the official bank builder is structurally dead in this configuration, not merely
    # sparse. The previous prompt did state the FIELD but never the POLICY ("what is worth
    # nominating"), so a general model had no reason to use it and defaulted to the safe no-op.
    #
    # This restores the channel by instruction instead of by training. It is a substitute for the
    # learned component, not an equivalent of it: a trained head is sensitive to state transitions
    # by construction, while a prompted model has to reason about them. Whether it does is
    # measured, not assumed -- `census_channel_b` reports `steps_with_planner_nomination` and the
    # census FAILs the arm when this knob is on and that count is zero.
    nomination_prompt: bool = False

    @property
    def enabled(self) -> bool:
        return self.stage_anchor or self.bank_max > 0


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_memory_system_config() -> MemorySystemConfig:
    return MemorySystemConfig(
        stage_anchor=_truthy(os.environ.get("MEM_STAGE_ANCHOR"), default=False),
        salience_subtask_change=_truthy(os.environ.get("MEM_SALIENCE_SUBTASK"), default=True),
        bank_max=int(os.environ.get("MEM_BANK_MAX", os.environ.get("K_MAX", "8"))),
        cluster_distance=int(os.environ.get("MEM_CLUSTER_D", os.environ.get("D_MERGE", "4"))),
        recent_window=int(os.environ.get("N_RECENT", "7")),
        kf_spread=int(os.environ.get("MEM_KF_SPREAD", "0")),
        nomination_prompt=_truthy(os.environ.get("MEM_KF_NOMINATION_PROMPT"), default=False),
    )
