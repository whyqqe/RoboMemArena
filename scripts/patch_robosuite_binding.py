#!/usr/bin/env python3
"""Patch robosuite binding_utils for mujoco>=3.1 numpy joint_type scalars."""

from __future__ import annotations

import sys
from pathlib import Path


def patch_binding_utils(path: Path) -> bool:
    text = path.read_text()
    old = "        joint_type = self.jnt_type[joint_id]\n"
    new = "        joint_type = int(self.jnt_type[joint_id])\n"
    if new in text:
        print(f"[OK] already patched: {path}")
        return True
    count = text.count(old)
    if count != 2:
        print(f"[ERR] expected 2 occurrences of joint_type assignment, found {count} in {path}", file=sys.stderr)
        return False
    path.write_text(text.replace(old, new))
    print(f"[OK] patched {path}")
    return True


def main() -> int:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        import site

        candidates = []
        for sp in site.getsitepackages():
            candidates.append(Path(sp) / "robosuite" / "utils" / "binding_utils.py")
        # venv layout: lib/python3.11/site-packages
        here = Path(__file__).resolve().parents[1]
        candidates.append(here / ".venv" / "lib" / "python3.11" / "site-packages" / "robosuite" / "utils" / "binding_utils.py")
        target = next((p for p in candidates if p.is_file()), None)
        if target is None:
            print("[ERR] binding_utils.py not found; pass path explicitly", file=sys.stderr)
            return 1
    return 0 if patch_binding_utils(target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
