"""Pytest plumbing.

Our scripts are named ``01_*.py``, ``02_*.py``, ... — leading digits make
them non-importable as regular modules. This conftest loads them via
``importlib`` and exposes each as a fixture.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str, filename: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def convert_mod():
    return _load("convert_mod", "02_convert_labelme_to_yolo.py")


@pytest.fixture(scope="session")
def merge_mod():
    return _load("merge_mod", "03_merge_and_version.py")
