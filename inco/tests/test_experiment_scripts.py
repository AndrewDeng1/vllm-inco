# SPDX-License-Identifier: Apache-2.0
"""The README's commands and `scripts/experiments/` must not drift apart.

Every experiment in the README is a script; a renamed or deleted script that
the README still tells people to run is the failure this guards against.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

INCO_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = INCO_DIR / "scripts" / "experiments"
README = INCO_DIR / "README.md"

SCRIPTS = sorted(p for p in EXP_DIR.glob("*.sh") if p.name != "_env.sh")
README_REFS = sorted(
    set(re.findall(r"inco/scripts/experiments/([\w.-]+\.sh)", README.read_text()))
)


def test_scripts_exist():
    assert SCRIPTS, f"no experiment scripts in {EXP_DIR}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_executable(script):
    assert script.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_sources_shared_env(script):
    """Paths come from _env.sh, so no script hardcodes a repo location."""
    assert '/_env.sh"' in script.read_text()


@pytest.mark.parametrize("name", README_REFS)
def test_readme_reference_resolves(name):
    assert (EXP_DIR / name).is_file(), f"README runs {name}, which does not exist"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_documented(script):
    assert script.name in README_REFS, f"{script.name} is not in the README"


def test_env_resolves_repo_paths(tmp_path):
    """_env.sh must resolve REPO from its own location, not the caller's cwd."""
    out = subprocess.run(
        ["bash", "-c", f'source "{EXP_DIR / "_env.sh"}" && echo "$REPO" && echo "$M"'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    repo, modal = Path(out[0]), Path(out[1])
    assert repo == INCO_DIR.parent
    assert modal == repo / ".venv" / "bin" / "modal"


def test_env_is_not_executable():
    """_env.sh is sourced; marking it runnable invites running it."""
    assert not os.access(EXP_DIR / "_env.sh", os.X_OK)
