"""Compatibility shim for environments where stdlib `pipes` is removed (Python 3.13)."""

from shlex import quote

__all__ = ["quote"]
