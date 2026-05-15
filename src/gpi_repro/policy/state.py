import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_wxyz_to_rotation(quat):
    quat = normalize_quat(quat)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def quat_xyzw_to_rotation(quat):
    quat = normalize_quat(quat)
    return Rotation.from_quat(quat)


def slerp_rotations(start_rot, end_rot, blend):
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return start_rot
    if blend >= 1.0:
        return end_rot
    key_rots = Rotation.from_quat(np.stack([
        start_rot.as_quat(),
        end_rot.as_quat(),
    ]))
    return Slerp([0.0, 1.0], key_rots)([blend])[0]


def quat_to_rotation(quat, quat_format):
    quat_format = str(quat_format).lower()
    if quat_format == "wxyz":
        return quat_wxyz_to_rotation(quat)
    if quat_format == "xyzw":
        return quat_xyzw_to_rotation(quat)
    raise ValueError(f"Unknown pose_quat_format: {quat_format}")


def rotation_to_quat(rot, quat_format):
    quat_xyzw = rot.as_quat()
    quat_format = str(quat_format).lower()
    if quat_format == "wxyz":
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
    if quat_format == "xyzw":
        return quat_xyzw
    raise ValueError(f"Unknown pose_quat_format: {quat_format}")


def relative_state_from_values(
    object_data,
    eef_pos,
    eef_quat,
    gripper_qpos,
    object_pose_start,
    target_pos=None,
    pose_quat_format="xyzw",
    relative_pose_mode="eef_frame_object_delta",
):
    object_data = np.asarray(object_data)
    object_pos = object_data[object_pose_start:object_pose_start + 3]
    object_quat = object_data[object_pose_start + 3:object_pose_start + 7]
    if len(object_quat) < 4:
        object_quat = np.array([1.0, 0.0, 0.0, 0.0])
    if target_pos is None:
        object_to_target_pos = np.zeros(3, dtype=np.float64)
    else:
        object_to_target_pos = object_pos - np.asarray(target_pos, dtype=np.float64)

    object_quat = normalize_quat(object_quat)
    eef_quat = normalize_quat(eef_quat)

    obj_rot = quat_to_rotation(object_quat, pose_quat_format)
    eef_rot = quat_to_rotation(eef_quat, pose_quat_format)
    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    if str(relative_pose_mode).lower() == "world_eef_minus_object":
        relative_pos = eef_pos - object_pos
        relative_rot = obj_rot.inv() * eef_rot
    else:
        object_to_eef_delta_world = object_pos - eef_pos
        relative_pos = eef_rot.inv().apply(object_to_eef_delta_world)
        relative_rot = eef_rot.inv() * obj_rot
    object_frame_relative_pos = obj_rot.inv().apply(eef_pos - object_pos)
    relative_quat = rotation_to_quat(relative_rot, pose_quat_format)

    return {
        "object_pos": object_pos.copy(),
        "object_quat": object_quat.copy(),
        "relative_pos": relative_pos,
        "relative_quat": normalize_quat(relative_quat),
        "robot0_gripper_qpos": np.asarray(gripper_qpos),
        "object_frame_relative_pos": object_frame_relative_pos,
        "object_z": np.array([object_pos[2]], dtype=np.float64),
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": normalize_quat(eef_quat),
        "object_to_target_pos": object_to_target_pos,
        "object_quat": object_quat.copy(),
    }


def state_from_demo_obs(
    obs_dict,
    timestep,
    object_pose_start,
    target_pos=None,
    pose_quat_format="xyzw",
    relative_pose_mode="eef_frame_object_delta",
):
    return relative_state_from_values(
        object_data=obs_dict["object"][timestep],
        eef_pos=obs_dict["robot0_eef_pos"][timestep],
        eef_quat=obs_dict["robot0_eef_quat"][timestep],
        gripper_qpos=obs_dict["robot0_gripper_qpos"][timestep],
        object_pose_start=object_pose_start,
        target_pos=target_pos,
        pose_quat_format=pose_quat_format,
        relative_pose_mode=relative_pose_mode,
    )


def state_from_live_obs(
    obs,
    object_pose_start,
    target_pos=None,
    pose_quat_format="xyzw",
    relative_pose_mode="eef_frame_object_delta",
):
    return relative_state_from_values(
        object_data=np.asarray(obs["object"]),
        eef_pos=np.asarray(obs["robot0_eef_pos"]),
        eef_quat=np.asarray(obs["robot0_eef_quat"]),
        gripper_qpos=np.asarray(obs["robot0_gripper_qpos"]),
        object_pose_start=object_pose_start,
        target_pos=target_pos,
        pose_quat_format=pose_quat_format,
        relative_pose_mode=relative_pose_mode,
    )


def set_abs_controller(env_meta):
    controller_config = env_meta["env_kwargs"].get("controller_configs")
    if not controller_config:
        return
    if controller_config.get("type") == "BASIC" and "body_parts" in controller_config:
        for body_part_config in controller_config["body_parts"].values():
            if isinstance(body_part_config, dict):
                body_part_config["control_delta"] = False
    else:
        controller_config["control_delta"] = False


def set_delta_controller(env_meta):
    controller_config = env_meta["env_kwargs"].get("controller_configs")
    if not controller_config:
        return
    if controller_config.get("type") == "BASIC" and "body_parts" in controller_config:
        for body_part_config in controller_config["body_parts"].values():
            if isinstance(body_part_config, dict):
                body_part_config["control_delta"] = True
    else:
        controller_config["control_delta"] = True


def patch_egl_probe():
    import sys
    from unittest.mock import MagicMock

    sys.modules["egl_probe"] = MagicMock()
    sys.modules["egl_probe"].get_available_devices = MagicMock(return_value=[0])


def maybe_unwrap_basic_for_robosuite_14(env_meta):
    try:
        import robosuite
        version_parts = tuple(
            int(part) for part in robosuite.__version__.split(".")[:2]
            if part.isdigit()
        )
    except Exception:
        version_parts = (1, 4)
    controller_config = env_meta["env_kwargs"].get("controller_configs")
    if (
        version_parts < (1, 5)
        and isinstance(controller_config, dict)
        and controller_config.get("type") == "BASIC"
        and "body_parts" in controller_config
    ):
        right_config = controller_config["body_parts"]["right"]
        if right_config.get("interpolation") is None:
            right_config["interpolation"] = "none"
        env_meta["env_kwargs"]["controller_configs"] = right_config
    elif (
        version_parts >= (1, 5)
        and isinstance(controller_config, dict)
        and controller_config.get("type") != "BASIC"
        and "body_parts" not in controller_config
    ):
        if controller_config.get("interpolation") is None:
            controller_config["interpolation"] = "none"
        if "damping" in controller_config and "damping_ratio" not in controller_config:
            controller_config["damping_ratio"] = controller_config["damping"]
        if "damping_limits" in controller_config and "damping_ratio_limits" not in controller_config:
            controller_config["damping_ratio_limits"] = controller_config["damping_limits"]
        control_delta = bool(controller_config.get("control_delta", True))
        controller_config["input_type"] = "delta" if control_delta else "absolute"
        controller_config["input_ref_frame"] = "base" if control_delta else "world"
        controller_config.setdefault("gripper", {"type": "GRIP"})
        env_meta["env_kwargs"]["controller_configs"] = {
            "type": "BASIC",
            "body_parts": {"right": controller_config},
        }


def live_object_pose_start(task=None):
    if task == "lift":
        return 0
    try:
        import robosuite
        version_parts = tuple(
            int(part) for part in robosuite.__version__.split(".")[:2]
            if part.isdigit()
        )
    except Exception:
        version_parts = (1, 4)
    return 7 if version_parts >= (1, 5) else 0


def first_sustained_true(mask, sustain_steps=1):
    mask = np.asarray(mask, dtype=bool)
    sustain_steps = max(1, int(sustain_steps))
    if sustain_steps <= 1:
        indices = np.flatnonzero(mask)
        return None if len(indices) == 0 else int(indices[0])
    if len(mask) < sustain_steps:
        return None
    hits = np.convolve(mask.astype(np.int32), np.ones(sustain_steps, dtype=np.int32), mode="valid")
    indices = np.flatnonzero(hits >= sustain_steps)
    return None if len(indices) == 0 else int(indices[0])
