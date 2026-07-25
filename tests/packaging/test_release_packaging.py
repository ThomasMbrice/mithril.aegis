"""
Release / packaging tests (REL-*).

These guard the **library-only distribution decision** and turn the ad-hoc
release-verification shell checks into regression tests, so that:

  - a stray re-addition of ``[project.scripts]`` fails CI,
  - a rename/removal in the public API surface fails CI,
  - the documented quickstart lifecycle (lib.md §3) keeps working.

AEGIS ships as an in-process library with **no** ``aegis`` console script
(see the comment in ``pyproject.toml`` and lib.md §9). The current
``aegis/cli.py`` remains only as an experimental torchrun stub and must not be
wired up as an entry point.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import tomllib
from pathlib import Path

import pytest

import aegis

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def reset_aegis():
    """AEGIS is a process-global singleton — reset around every test."""
    aegis._reset()
    yield
    aegis._reset()


def _pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


# ---------------------------------------------------------------------------
# Packaging invariants — the "library-only" decision


def test_no_console_script_in_installed_metadata():
    """The installed distribution must expose no console_scripts."""
    eps = importlib_metadata.distribution("aegis").entry_points
    console_scripts = sorted(e.name for e in eps if e.group == "console_scripts")
    assert console_scripts == [], (
        "AEGIS ships library-only; unexpected console scripts in installed "
        f"metadata: {console_scripts}"
    )


def test_pyproject_declares_no_scripts():
    """pyproject.toml must not declare [project.scripts]."""
    project = _pyproject().get("project", {})
    assert "scripts" not in project, (
        "pyproject.toml declares [project.scripts]; the CLI was intentionally "
        "dropped for the library-only release (see pyproject comment / lib.md §9)."
    )


def test_version_matches_pyproject():
    """aegis.__version__ must stay in sync with the packaged version."""
    assert aegis.__version__ == _pyproject()["project"]["version"]


def test_cli_module_present_but_unwired():
    """cli.py stays as an experimental stub — importable, but not an entry point."""
    import aegis.cli as cli

    assert hasattr(cli, "main"), "aegis.cli.main should still exist as a stub"


# ---------------------------------------------------------------------------
# Public API surface — the library's contract


@pytest.mark.parametrize(
    "name",
    ["init", "status", "explain", "dashboard", "disable",
     "checkpoint", "transport", "policy"],
)
def test_public_api_present(name: str):
    assert hasattr(aegis, name), f"aegis.{name} missing from the public API"


def test_explicit_apis_require_init():
    """Every explicit control API must raise before init() — no silent no-op."""
    with pytest.raises(RuntimeError):
        aegis.checkpoint.restore()
    with pytest.raises(RuntimeError):
        aegis.transport.get_fast_path()
    with pytest.raises(RuntimeError):
        aegis.policy.set("economics.policy", "correctness_first")
    with pytest.raises(RuntimeError):
        aegis.status()


# ---------------------------------------------------------------------------
# Quickstart lifecycle (lib.md §3): init -> introspect -> disable


def test_quickstart_lifecycle():
    aegis.init()

    s = aegis.status()
    assert s["initialized"] is True
    assert s["mode"] == "active"
    assert set(s["active_hooks"]) == {
        "transport", "compute", "checkpoint", "telemetry", "policy",
    }

    # Dashboard renders in both formats.
    text = aegis.dashboard()
    assert isinstance(text, str) and text
    assert isinstance(aegis.dashboard(fmt="json"), dict)

    # explain() before any fault returns the placeholder message.
    assert "message" in aegis.explain()

    # Live policy update on the running job.
    aegis.policy.set("economics.policy", "correctness_first")

    # Kill switch — after disable, the runtime is gone and status() raises.
    aegis.disable()
    with pytest.raises(RuntimeError):
        aegis.status()
