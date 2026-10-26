"""Tunables for the vision perception pipeline.

Defaults follow the reference implementations we read before writing this:
SAM3's own image processor (score = class prob x presence prob), ReKep's
mask filtering, OK-Robot's mask-filtered grasps and cuRobo's robot
self-filter. Values that differ from those sources were measured on LIBERO
renders (outputs/perception_probe_*).
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIVILEGED = "privileged"
VISION = "vision"
PERCEPTION_MODES = (PRIVILEGED, VISION)


@dataclass
class PerceptionConfig:
    # agentview is the System 2 view; the wrist camera fills in what it occludes.
    camera_names: list[str] = field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    # 512 rather than the 256 policy-input size: thin drawer handles and
    # ramekin rims are only a few pixels wide at 256.
    render_height: int = 512
    render_width: int = 512
    depth_max_m: float = 3.0

    # SAM3 drops low-confidence instances itself; small LIBERO objects score
    # 0.3-0.6 even when the mask is clean, hence lower than SAM3's 0.5 default.
    sam3_score_threshold: float = 0.3
    # Overlapping instances from different category prompts are one object.
    mask_nms_iou: float = 0.5
    min_mask_area_frac: float = 0.0005
    max_mask_area_frac: float = 0.5
    # Depth at mask boundaries mixes foreground and background.
    mask_erode_px: int = 2

    voxel_size_m: float = 0.005
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    cluster_eps_m: float = 0.015
    cluster_min_points: int = 10

    # Depth points this close to the robot's own collision geometry are the
    # robot, not the scene (cuRobo's RobotSegmenter uses 5 cm).
    robot_filter_margin_m: float = 0.03

    table_ransac_threshold_m: float = 0.005
    table_ransac_iterations: int = 1000
    # Points this close above the table plane are the table, not the object.
    table_clearance_m: float = 0.004
    # Inflation applied to every perceived collision hull.
    hull_padding_m: float = 0.005

    # Association gate between a detection and an existing track.
    track_gate_m: float = 0.08
    # Re-run full perception every N env steps between event keyframes;
    # 0 disables periodic refresh (events still trigger one).
    refresh_every_steps: int = 0
    # Finger opening (m) that proves a grasp on its own. An empty gripper
    # creeps from ~5 mm to ~1 mm over 30 steps after closing, so thinner
    # grasps (a 1.4 mm bowl rim) are confirmed by lifting and looking instead.
    grasp_min_width_m: float = 0.004
    # A width-confirmed grasp whose fingers close below this has slipped.
    grasp_slip_width_m: float = 0.002
    # Max TCP distance to a perceived object's hull for it to count as the
    # thing being held when the gripper closes.
    attach_max_dist_m: float = 0.05

    drawer_plane_threshold_m: float = 0.004
    # A drawer's travel is unknown from one image; assume it can slide out
    # this fraction of the carcass depth seen along the pull axis.
    drawer_travel_frac_of_depth: float = 0.8

    # Generic part queries the skill library operates on, always run in
    # addition to the VLM's category proposals.
    part_prompts: list[str] = field(default_factory=lambda: ["drawer handle"])
    # Plain "knob" misses LIBERO's stove knob entirely; the colour-qualified
    # prompt finds it.
    knob_prompts: list[str] = field(default_factory=lambda: ["knob", "black knob", "dial", "control knob"])
    # A knob's travel can't be seen; assume it turns right-handed about its
    # mounting surface's outward normal (LIBERO's stove turns on that way)
    # from where it was first seen. The span must stay under pi:
    # measure_knob_rotation reads the axis off the rotation between the two
    # range ends, which flips sign past half a turn.
    knob_range_rad: tuple[float, float] = (-0.05, 2.6)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "PerceptionConfig":
        cfg = cls()
        for key, value in (raw or {}).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
