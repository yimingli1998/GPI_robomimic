import json
import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


import robomimic.utils.obs_utils as ObsUtils


OBS_KEYS = [
    "relative_pos",
    "relative_quat",
    "robot0_gripper_qpos",
    "object_frame_relative_pos",
    "object_z",
    "object_pos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "object_to_target_pos",
    "object_quat",
]
RAW_OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
QUATERNION_FEATURES = {"relative_quat", "robot0_eef_quat", "object_quat"}

TASK_TARGET_POS = {
    # Robosuite PegsArena square peg body position.
    "square": [0.23, 0.10, 0.85],
}
SQUARE_PHASE_NAMES = ["pick", "lift", "transport", "pre_insert", "insert"]
SQUARE_PHASE_IDS = {name: idx for idx, name in enumerate(SQUARE_PHASE_NAMES)}
SQUARE_STAGE_NAMES = ["pick", "lift", "transport", "pre_insert", "insert", "task"]
CAN_PHASE_NAMES = ["pick", "lift", "transport"]
CAN_PHASE_IDS = {name: idx for idx, name in enumerate(CAN_PHASE_NAMES)}
CAN_STAGE_NAMES = ["pick", "lift", "transport", "task"]
LIFT_PHASE_NAMES = ["pick", "lift"]


@dataclass(frozen=True)
class TaskGPIProfile:
    """Task geometry used by the GPI-style retrieval and correction pipeline."""

    name: str
    phase_names: tuple

    @property
    def phase_ids(self):
        return {phase: idx for idx, phase in enumerate(self.phase_names)}


TASK_GPI_PROFILES = {
    "can": TaskGPIProfile(
        name="can",
        phase_names=tuple(CAN_PHASE_NAMES),
    ),
    "square": TaskGPIProfile(
        name="square",
        phase_names=tuple(SQUARE_PHASE_NAMES),
    ),
    "lift": TaskGPIProfile(
        name="lift",
        phase_names=tuple(LIFT_PHASE_NAMES),
    ),
}


def parse_seeds(seed_arg):
    if ".." in seed_arg:
        start, end = seed_arg.split("..", 1)
        return list(range(int(start), int(end) + 1))
    return [int(seed.strip()) for seed in seed_arg.split(",") if seed.strip()]


def load_json_arg(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    value = str(value)
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(value)
    if isinstance(payload, dict) and "distance_config" in payload:
        return payload["distance_config"]
    if isinstance(payload, dict) and "distance_overrides" in payload:
        return payload["distance_overrides"]
    return payload










def _coerce_config_value(value):
    if isinstance(value, dict):
        return {key: _coerce_config_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_coerce_config_value(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_config_value(item) for item in value]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def _merge_config_values(base_value, update_value):
    if isinstance(update_value, dict) and isinstance(base_value, dict):
        merged = deepcopy(base_value)
        for key, sub_value in update_value.items():
            merged[key] = _merge_config_values(merged.get(key), sub_value)
        return merged
    return _coerce_config_value(update_value)


def merge_nested(base, updates):
    if not updates:
        return base
    merged = deepcopy(base)
    normalized = {}
    for key, value in updates.items():
        if "." not in key:
            normalized[key] = value
            continue
        parts = key.split(".")
        current = normalized
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    for key, value in normalized.items():
        merged[key] = _merge_config_values(merged.get(key), value)
    return merged


def default_distance_config(task):
    pick_lift_task = task in {"can", "lift"}
    return {
        "feature_weights": {
            "relative_pos": 1.0,
            "relative_quat": 1.0,
            "robot0_gripper_qpos": 1.0,
            "object_frame_relative_pos": 1.0,
            "object_z": 1.0,
            "object_pos": 1.0,
            "robot0_eef_pos": 1.0,
            "robot0_eef_quat": 0.2,
            "object_to_target_pos": 1.0,
            "object_quat": 1.0,
        },
        "pose_quat_format": "wxyz" if task == "lift" else "xyzw",
        "relative_pose_mode": (
            "world_eef_minus_object" if task == "lift" else "eef_frame_object_delta"
        ),
        "task_target_pos": TASK_TARGET_POS.get(task),
        "square_phase_gate_metric": False,
        "square_phase_gate_mode": "hard",
        "square_phase_gate_allow_next": True,
        "square_phase_gate_next_window": None,
        "square_phase_gate_fallback_to_all": True,
        "square_phase_gate_soft_scale_top_n": 64,
        "square_phase_gate_soft_min_penalty": 0.005,
        "square_phase_gate_soft_current_penalty": 0.0,
        "square_phase_gate_soft_next_penalty": 0.0,
        "square_phase_gate_soft_next_step_penalty": 0.75,
        "square_phase_gate_soft_next_window": 24,
        "square_phase_gate_soft_next_max_progress": 3.0,
        "square_phase_gate_soft_hard_boundary": False,
        "square_phase_gate_soft_min_phase": "pick",
        "square_phase_gate_soft_max_phase": "insert",
        "square_phase_gate_soft_previous_penalty": 1.5,
        "square_phase_gate_soft_skip_penalty": 4.0,
        "future_progress_phase_overrides": {},
        "future_progress_terminal_xy_weight": 0.0,
        "future_progress_terminal_z_weight": 0.0,
        "future_progress_terminal_ori_weight": 0.0,
        "future_progress_terminal_eef_weight": 0.0,
        "future_progress_terminal_xy_scale": 0.04,
        "future_progress_terminal_z_scale": 0.04,
        "future_progress_terminal_ori_scale": 0.8,
        "future_progress_terminal_eef_scale": 0.08,
        "future_progress_terminal_z_target": None,
        "future_progress_terminal_eef_target": None,
        "square_phase_gate_pick_lift_delta": 0.03,
        "square_phase_gate_table_z_window": 40,
        "square_phase_gate_table_z_percentile": 5.0,
        "square_phase_gate_lift_sustain_steps": 3,
        "square_phase_gate_lift_start_backtrack": 4,
        "square_phase_gate_lift_start_contact_eef_object_max": 0.085,
        "square_phase_gate_lift_start_gripper_open_threshold": 0.035,
        "square_phase_gate_lift_start_gripper_close_threshold": 0.03,
        "square_phase_gate_lift_window": 24,
        "square_phase_gate_transport_xy_threshold": 0.16,
        "square_phase_gate_pre_insert_xy_threshold": 0.08,
        "square_phase_gate_insert_xy_threshold": 0.04,
        "square_phase_gate_insert_z_threshold": 0.94,
        "square_object_abs_retarget_source": "action_implied",
        "square_object_abs_retarget_blend": 0.5,
        "square_object_abs_orientation_blend": None,
        "square_object_abs_orientation_transport_blend": 0.5,
        "square_object_abs_orientation_pre_insert_blend": 0.5,
        "square_object_abs_orientation_insert_blend": 1.0,
        "square_demo_calibrate_lift_thresholds": True,
        "square_demo_lift_success_lift_margin": 0.002,
        "square_demo_lift_success_eef_margin": 0.005,
        "square_demo_lift_start_gripper_margin": 0.003,
        "square_demo_lift_start_contact_margin": 0.005,
        "square_demo_lift_start_sigma_margin": 0.25,
        "square_demo_calibrate_stage_thresholds": True,
        "square_demo_stage_transport_xy_margin": 0.005,
        "square_demo_stage_pre_insert_xy_margin": 0.003,
        "square_demo_stage_insert_xy_margin": 0.002,
        "square_demo_stage_insert_z_margin": 0.002,
        "square_lift_start_preference": False,
        "square_lift_start_preference_hard_gate": False,
        "square_lift_start_preference_boost": 0.7,
        "square_lift_start_preference_penalty": 0.0,
        "square_lift_start_preference_relative_sigma_max": 4.0,
        "square_lift_start_preference_gripper_sigma_max": 3.0,
        "square_lift_start_preference_min_std": 0.01,
        "square_lift_start_preference_window_after": 12,
        "square_pickup_eef_object_max": 0.11,
        "square_pick_lift_z_threshold": 0.92,
        "square_carry_eef_object_max": 0.11,
        "square_pre_insert_xy_threshold": 0.08,
        "square_insert_xy_threshold": 0.04,
        "square_insert_z_threshold": 0.94,
        "can_demo_target_from_demos": True,
        "can_demo_target_percentile": 50.0,
        "can_phase_gate_metric": False,
        "can_phase_gate_mode": "hard",
        "can_phase_gate_allow_next": True,
        "can_phase_gate_next_window": None,
        "can_phase_gate_fallback_to_all": True,
        "can_phase_gate_min_phase": "pick",
        "can_phase_gate_pick_lift_delta": 0.03,
        "can_phase_gate_table_z_window": 40,
        "can_phase_gate_table_z_percentile": 5.0,
        "can_phase_gate_lift_sustain_steps": 3,
        "can_phase_gate_lift_start_backtrack": 4,
        "can_phase_gate_lift_start_contact_eef_object_max": 0.085,
        "can_phase_gate_lift_start_gripper_open_threshold": 0.035,
        "can_phase_gate_lift_start_gripper_close_threshold": 0.03,
        "can_phase_gate_lift_window": 24,
        "can_lift_start_preference": False,
        "can_lift_start_preference_hard_gate": False,
        "can_lift_start_preference_boost": 0.7,
        "can_lift_start_preference_penalty": 0.0,
        "can_lift_start_preference_relative_sigma_max": 4.0,
        "can_lift_start_preference_gripper_sigma_max": 3.0,
        "can_lift_start_preference_min_std": 0.01,
        "can_lift_start_preference_window_after": 12,
        "can_lift_start_relaxed_ready": False,
        "can_lift_start_relaxed_gripper_max": 0.045,
        "can_lift_start_relaxed_contact_max": None,
        "can_lift_start_relaxed_ignore_gripper_profile": True,
        "can_demo_calibrate_lift_thresholds": True,
        "can_demo_lift_success_lift_margin": 0.002,
        "can_demo_lift_success_eef_margin": 0.005,
        "can_demo_lift_start_gripper_margin": 0.003,
        "can_demo_lift_start_contact_margin": 0.005,
        "can_demo_lift_start_sigma_margin": 0.25,
        "can_demo_calibrate_stage_thresholds": True,
        "can_demo_stage_transport_xy_percentile": 95.0,
        "can_demo_stage_transport_xy_margin": 0.010,
        "can_transport_success_xy_threshold": 0.08,
        "lift_phase_gate_metric": False,
        "lift_phase_gate_mode": "hard",
        "lift_phase_gate_allow_next": True,
        "lift_phase_gate_next_window": None,
        "lift_phase_gate_fallback_to_all": True,
        "lift_phase_gate_min_phase": "pick",
        "lift_phase_gate_pick_lift_delta": 0.03,
        "lift_phase_gate_table_z_window": 40,
        "lift_phase_gate_table_z_percentile": 5.0,
        "lift_phase_gate_lift_sustain_steps": 3,
        "lift_phase_gate_lift_start_backtrack": 4,
        "lift_phase_gate_lift_start_contact_eef_object_max": 0.085,
        "lift_phase_gate_lift_start_gripper_open_threshold": 0.035,
        "lift_phase_gate_lift_start_gripper_close_threshold": 0.03,
        "lift_phase_gate_lift_window": 24,
        "lift_lift_start_preference": False,
        "lift_lift_start_preference_hard_gate": False,
        "lift_lift_start_preference_boost": 0.7,
        "lift_lift_start_preference_penalty": 0.0,
        "lift_lift_start_preference_relative_sigma_max": 4.0,
        "lift_lift_start_preference_gripper_sigma_max": 3.0,
        "lift_lift_start_preference_min_std": 0.01,
        "lift_lift_start_preference_window_after": 12,
        "lift_lift_start_relaxed_ready": False,
        "lift_lift_start_relaxed_gripper_max": 0.045,
        "lift_lift_start_relaxed_contact_max": None,
        "lift_lift_start_relaxed_ignore_gripper_profile": True,
        "lift_demo_calibrate_lift_thresholds": True,
        "lift_demo_lift_success_lift_margin": 0.002,
        "lift_demo_lift_success_eef_margin": 0.005,
        "lift_demo_lift_start_gripper_margin": 0.003,
        "lift_demo_lift_start_contact_margin": 0.005,
        "lift_demo_lift_start_sigma_margin": 0.25,
    }


def target_pos_from_config(task, distance_config=None):
    distance_config = distance_config or {}
    target_pos = distance_config.get("task_target_pos", TASK_TARGET_POS.get(task))
    if target_pos is None:
        return None
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)
    if target_pos.shape[0] != 3:
        raise ValueError(f"task_target_pos must have length 3, got {target_pos.shape[0]}")
    return target_pos
