"""Thin sitecustomize entrypoint — keeps activation consistent with sibling arms."""
from __future__ import annotations


def install() -> None:
    from dial.bind import install as _install
    _install()
