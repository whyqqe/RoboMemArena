from __future__ import annotations

from typing import Any


def release_gripper(env: Any, *, steps: int = 8, open_value: float = -1.0) -> None:
    """Minimal analytic primitive inspired by HarnessVLA `release` / `set_gripper`."""
    action = [0.0] * 6 + [float(open_value)]
    for _ in range(max(1, steps)):
        env.step(action)
