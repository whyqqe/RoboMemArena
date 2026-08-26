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
    )
