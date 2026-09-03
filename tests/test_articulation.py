"""Tests for drawer/cabinet fixture discovery and measurement
(step/system_one/motion/articulation.py) against the real LIBERO
"open the top drawer and put the bowl inside" scene."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_libero_drawer

from step.system_one.motion.articulation import (
    discover_drawers,
    locate_handle_position_world,
    measure_pull_axis_and_travel,
    resolve_drawer,
)


@requires_libero_drawer
def test_discover_drawers_finds_wooden_cabinet_levels(libero_drawer_scene):
    drawers = discover_drawers(libero_drawer_scene)
    levels = {d.level_label for d in drawers}
    assert "top" in levels
    # The wooden_cabinet asset has three slide-jointed levels; every real
    # discovered drawer must carry a resolvable joint and at least one geom.
    for d in drawers:
        assert libero_drawer_scene.joint_id(d.joint_name) >= 0
        assert libero_drawer_scene.body_geom_ids(d.drawer_body, include_children=False)


@requires_libero_drawer
@pytest.mark.parametrize("query", ["top_drawer", "top_drawer_handle", "top drawer"])
def test_resolve_drawer_matches_top_level_by_various_names(libero_drawer_scene, query):
    resolved = resolve_drawer(libero_drawer_scene, query)
    assert resolved.level_label == "top"


@requires_libero_drawer
def test_resolve_drawer_matches_raw_joint_name(libero_drawer_scene):
    drawers = discover_drawers(libero_drawer_scene)
    top = next(d for d in drawers if d.level_label == "top")
    resolved = resolve_drawer(libero_drawer_scene, top.joint_name)
    assert resolved.joint_name == top.joint_name


@requires_libero_drawer
def test_resolve_drawer_raises_with_available_list_on_no_match(libero_drawer_scene):
    with pytest.raises(KeyError):
        resolve_drawer(libero_drawer_scene, "nonexistent_fixture_xyz")


@requires_libero_drawer
def test_measure_pull_axis_and_travel_is_horizontal_and_matches_known_range(libero_drawer_scene):
    top = resolve_drawer(libero_drawer_scene, "top_drawer")
    pull_axis_world, travel_m, qpos_closed, qpos_open = measure_pull_axis_and_travel(
        libero_drawer_scene, top
    )
    assert np.isclose(np.linalg.norm(pull_axis_world), 1.0, atol=1e-6)
    # A drawer slides out roughly horizontally, not vertically.
    assert abs(pull_axis_world[2]) < 0.2
    # Matches the wooden_cabinet.xml joint range [-0.16, 0.01] (~0.17m travel).
    assert 0.1 < travel_m < 0.25
    assert not np.isclose(qpos_closed, qpos_open)


@requires_libero_drawer
def test_locate_handle_position_is_in_front_of_drawer_body_not_behind_it(libero_drawer_scene):
    """Regression: an earlier version used the joint's own named '_region'
    site as the handle anchor, which turned out to be LIBERO's *interior*
    placement-goal region (behind the front face, unreachable/nonsensical as
    a grasp target) rather than the handle. The handle must sit further
    toward the open direction than the drawer body's own origin."""
    top = resolve_drawer(libero_drawer_scene, "top_drawer")
    pull_axis_world, _, _, _ = measure_pull_axis_and_travel(libero_drawer_scene, top)
    handle_pos = locate_handle_position_world(libero_drawer_scene, top, pull_axis_world)
    body_pos, _ = libero_drawer_scene.body_pose(top.drawer_body)
    assert np.dot(handle_pos, pull_axis_world) > np.dot(body_pos, pull_axis_world)
