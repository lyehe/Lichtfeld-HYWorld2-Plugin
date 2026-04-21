"""Inject Visual Studio build env vars into the current process on Windows.

torch.compile on Windows uses the Inductor backend, which generates C
code and shells out to MSVC ``cl.exe`` to compile it. When Python is
launched from a GUI app (LFS), the Visual Studio *Developer Command
Prompt* env vars (``INCLUDE``, ``LIB``, ``LIBPATH``, augmented ``PATH``)
are not inherited, so ``cl.exe`` fires but can't find ``stdio.h`` /
``windows.h`` / the runtime libs, and Inductor crashes with
``InductorError: CalledProcessError``.

This module detects ``vcvarsall.bat`` under a standard VS 2022 install,
runs it in a subprocess (as a no-op stdout), and merges only the build-
related env deltas (``INCLUDE``, ``LIB``, ``LIBPATH``, ``PATH``) into
``os.environ``. Plugin-local settings (``HF_HOME``, ``CUDA_HOME`` on
Linux, etc.) are left untouched.

Idempotent — only applies once per process. No-op off Windows.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_APPLIED: bool | None = None  # None = not tried, True = applied, False = failed
_REASON: str = ""


def _find_vcvarsall() -> Path | None:
    if sys.platform != "win32":
        return None
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022"),
    ]
    editions = ("Community", "Professional", "Enterprise", "BuildTools")
    for root in roots:
        if not root.is_dir():
            continue
        for edition in editions:
            p = root / edition / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if p.is_file():
                return p
    return None


def apply() -> tuple[bool, str]:
    """Ensure MSVC build env vars are present in os.environ.

    Returns (applied, reason). ``applied`` is True when the env is
    either now active or was already usable. ``reason`` is a short
    human-readable diagnostic for logs.
    """
    global _APPLIED, _REASON
    if _APPLIED is True:
        return True, "already applied"
    if _APPLIED is False:
        return False, _REASON
    if sys.platform != "win32":
        _APPLIED, _REASON = False, "not windows"
        return False, _REASON
    # If the caller already has a dev-prompt env, don't re-run.
    if os.environ.get("INCLUDE") and os.environ.get("LIB"):
        _APPLIED, _REASON = True, "inherited from parent process"
        return True, _REASON
    vcvars = _find_vcvarsall()
    if vcvars is None:
        _APPLIED, _REASON = False, "vcvarsall.bat not found under 'Program Files[ (x86)]/Microsoft Visual Studio/2022/*'"
        return False, _REASON
    try:
        out = subprocess.check_output(
            f'"{vcvars}" x64 >nul && set',
            shell=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        _APPLIED, _REASON = False, f"vcvarsall invocation failed: {exc}"
        return False, _REASON
    wanted = ("INCLUDE", "LIB", "LIBPATH", "PATH")
    touched: list[str] = []
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if not sep or k.upper() not in wanted:
            continue
        os.environ[k] = v
        touched.append(k)
    _APPLIED = bool(touched)
    _REASON = f"applied: {', '.join(sorted(set(touched)))}" if touched else "vcvarsall emitted no relevant vars"
    return _APPLIED, _REASON
