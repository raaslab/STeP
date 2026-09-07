"""Shared fixtures for the motion-pipeline test suite.

Tests that need a real LIBERO/MuJoCo scene are marked ``@pytest.mark.libero``
and use the session-scoped ``libero_scene`` fixture (constructing the
robosuite/LIBERO env is the expensive part, ~1-2s, so it's built once per test
session and every test gets a fresh `.clone()`d scratch SceneQuery -- the
shared live env itself is never mutated by any test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STEP_ROOT = Path(__file__).resolve().parents[1]
VLM_STL_ROOT = (STEP_ROOT / ".." / "vlm-stl").resolve()

for _candidate in (VLM_STL_ROOT / "LIBERO", VLM_STL_ROOT):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
if str(STEP_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP_ROOT))

from step.system_one.libero_scene import _import_libero_benchmark, _import_libero_env  # noqa: E402
from step.system_one.motion.panda import (  # noqa: E402
    EE_SITE,
    GRIPPER_BODIES,
    HAND_SITE,
    JOINT_NAMES,
    ROBOT_BODIES,
)


def _bddl_path(suite: str, filename: str) -> Path | None:
    """Resolve a bundled bddl file through the active ``libero`` install
    (whatever its package layout -- see ``_import_libero_benchmark``)
    instead of a hardcoded vlm-stl-relative path, so this also works
    against a normal pip/uv-installed LIBERO (e.g. Cap-X's
    ``.venv-libero``) that isn't checked out next to this repo at all."""
    try:
        _, get_libero_path = _import_libero_benchmark()
        return Path(get_libero_path("bddl_files")) / suite / filename
    except Exception:
        return None


ALPHABET_SOUP_BDDL = _bddl_path("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl")
OPEN_DRAWER_BDDL = _bddl_path("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside.bddl")


def _libero_available() -> bool:
    if ALPHABET_SOUP_BDDL is None:
        return False
    try:
        _import_libero_env()
    except Exception:
        return False
    return ALPHABET_SOUP_BDDL.exists()


requires_libero = pytest.mark.skipif(not _libero_available(), reason="LIBERO/robosuite env not available")


@pytest.fixture(scope="session")
def libero_env():
    if not _libero_available():
        pytest.skip("LIBERO/robosuite env not available")
    OffScreenRenderEnv = _import_libero_env()

    env = OffScreenRenderEnv(bddl_file_name=str(ALPHABET_SOUP_BDDL), camera_heights=64, camera_widths=64)
    env.reset()
    yield env
    env.close()


@pytest.fixture()
def libero_scene(libero_env):
    """A fresh scratch SceneQuery for each test, synced from the shared live
    env but never written back to it."""
    from step.system_one.motion.scene_query import SceneQuery

    return SceneQuery.from_robosuite_sim(libero_env.env.sim)


def _libero_drawer_available() -> bool:
    if OPEN_DRAWER_BDDL is None:
        return False
    try:
        _import_libero_env()
    except Exception:
        return False
    return OPEN_DRAWER_BDDL.exists()


requires_libero_drawer = pytest.mark.skipif(
    not _libero_drawer_available(), reason="LIBERO/robosuite drawer env not available"
)


@pytest.fixture(scope="session")
def libero_drawer_env():
    if not _libero_drawer_available():
        pytest.skip("LIBERO/robosuite drawer env not available")
    OffScreenRenderEnv = _import_libero_env()

    env = OffScreenRenderEnv(bddl_file_name=str(OPEN_DRAWER_BDDL), camera_heights=64, camera_widths=64)
    env.reset()
    yield env
    env.close()


@pytest.fixture()
def libero_drawer_scene(libero_drawer_env):
    """A fresh scratch SceneQuery for each test, synced from the shared live
    drawer-task env but never written back to it."""
    from step.system_one.motion.scene_query import SceneQuery

    return SceneQuery.from_robosuite_sim(libero_drawer_env.env.sim)
