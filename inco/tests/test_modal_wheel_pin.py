# SPDX-License-Identifier: Apache-2.0
"""The Modal images must pin a wheel that actually exists.

`origin` is the fork, so every commit on its `main` lacks a nightly wheel.
Resolving the pin has to walk back to an upstream ancestor instead of failing,
which is the regression these tests hold in place.
"""

import importlib.util
import io
import json
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("modal")

MODAL_DIR = Path(__file__).resolve().parent.parent / "modal"
MODULES = ["modal_baseline.py", "modal_speculators.py"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _load(name, monkeypatch, wheels, probes=None):
    """Import a modal app with a fake wheel index. `wheels`: commits that have one."""

    def fake_urlopen(url, timeout=None):
        commit = url.split("/")[3]
        if probes is not None:
            probes.append(commit)
        if commit not in wheels:
            raise urllib.request.HTTPError(url, 404, "Not Found", {}, None)
        return _Resp(json.dumps([{"version": f"0.0.0+g{commit[:9]}"}]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    spec = importlib.util.spec_from_file_location(f"_t_{name}", MODAL_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ancestors(n=20):
    import subprocess

    repo = str(MODAL_DIR.parent.parent)
    out = subprocess.check_output(
        ["git", "-C", repo, "rev-list", f"--max-count={n}", "HEAD"], text=True
    )
    return out.split()


@pytest.mark.parametrize("name", MODULES)
def test_walks_back_to_an_ancestor_with_a_wheel(name, monkeypatch):
    """HEAD has no nightly; the pin lands on the nearest ancestor that does."""
    history = _ancestors()
    target = history[6]
    probes = []
    module = _load(name, monkeypatch, {target}, probes)

    assert module.FORK_BUILD_ENV["VLLM_PRECOMPILED_WHEEL_COMMIT"] == target
    assert probes == history[:7], "must probe newest-first and stop at the first hit"


@pytest.mark.parametrize("name", MODULES)
def test_explicit_pin_skips_the_walk(name, monkeypatch):
    history = _ancestors()
    pinned = history[3]
    monkeypatch.setenv("INCO_WHEEL_COMMIT", pinned)
    probes = []
    module = _load(name, monkeypatch, set(history), probes)

    assert module.FORK_BUILD_ENV["VLLM_PRECOMPILED_WHEEL_COMMIT"] == pinned
    assert probes == [pinned]


@pytest.mark.parametrize("name", MODULES)
def test_no_wheel_anywhere_raises(name, monkeypatch):
    monkeypatch.setenv("INCO_WHEEL_SEARCH_DEPTH", "5")
    with pytest.raises(RuntimeError, match="no cu130 nightly wheel"):
        _load(name, monkeypatch, set())


@pytest.mark.parametrize("name", MODULES)
def test_scm_version_matches_the_pinned_wheel(name, monkeypatch):
    """setuptools-scm cannot see .git, so the reported version must be the
    wheel's own -- otherwise the install claims kernels it is not linking."""
    target = _ancestors()[2]
    module = _load(name, monkeypatch, {target})
    env = module.FORK_BUILD_ENV

    assert env["SETUPTOOLS_SCM_PRETEND_VERSION"] == f"0.0.0+g{target[:9]}"
    assert env["VLLM_MAIN_CUDA_VERSION"] == "13.0"
