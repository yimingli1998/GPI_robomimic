import numpy as np

from .config import CAN_PHASE_IDS, CAN_STAGE_NAMES, SQUARE_PHASE_IDS, SQUARE_STAGE_NAMES
from .state import state_from_live_obs


def pick_success_info(
    obs,
    object_pose_start,
    initial_object_z,
    table_object_z,
    lift_delta,
    eef_object_max,
    baseline="table_min",
):
    object_data = np.asarray(obs["object"])
    object_pos = object_data[object_pose_start:object_pose_start + 3]
    eef_pos = np.asarray(obs["robot0_eef_pos"])
    if baseline == "initial":
        lift_baseline = float(initial_object_z)
    elif baseline == "table_min":
        lift_baseline = float(table_object_z)
    else:
        raise ValueError(f"Unknown pick success baseline: {baseline}")
    object_lift = float(object_pos[2] - lift_baseline)
    object_lift_from_initial = float(object_pos[2] - initial_object_z)
    object_lift_from_table = float(object_pos[2] - table_object_z)
    eef_object_dist = float(np.linalg.norm(eef_pos - object_pos))
    success = object_lift >= float(lift_delta) and eef_object_dist <= float(eef_object_max)
    return {
        "success": bool(success),
        "object_z": float(object_pos[2]),
        "initial_object_z": float(initial_object_z),
        "table_object_z": float(table_object_z),
        "baseline": str(baseline),
        "lift_baseline": float(lift_baseline),
        "object_lift": float(object_lift),
        "object_lift_from_initial": float(object_lift_from_initial),
        "object_lift_from_table": float(object_lift_from_table),
        "eef_object_dist": float(eef_object_dist),
        "lift_delta": float(lift_delta),
        "eef_object_max": float(eef_object_max),
    }


def square_step_diagnostics(obs, policy):
    current_state = state_from_live_obs(
        obs,
        policy.object_pose_start,
        target_pos=policy.task_target_pos,
        pose_quat_format=policy.pose_quat_format,
        relative_pose_mode=policy.relative_pose_mode,
    )
    object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
    eef_pos = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
    object_to_target = np.asarray(
        current_state["object_to_target_pos"],
        dtype=np.float64,
    )
    relative_pos = np.asarray(current_state["relative_pos"], dtype=np.float64)
    object_frame_relative_pos = np.asarray(
        current_state["object_frame_relative_pos"],
        dtype=np.float64,
    )
    return {
        "object_z": float(object_pos[2]),
        "object_to_target_xy": float(np.linalg.norm(object_to_target[:2])),
        "object_to_target_z": float(object_to_target[2]),
        "object_to_target_pos": object_to_target.astype(float).tolist(),
        "object_pos": object_pos.astype(float).tolist(),
        "object_quat": np.asarray(
            current_state["object_quat"],
            dtype=np.float64,
        ).astype(float).tolist(),
        "eef_pos": eef_pos.astype(float).tolist(),
        "eef_quat": np.asarray(
            current_state["robot0_eef_quat"],
            dtype=np.float64,
        ).astype(float).tolist(),
        "relative_pos": relative_pos.astype(float).tolist(),
        "relative_pos_norm": float(np.linalg.norm(relative_pos)),
        "relative_quat": np.asarray(
            current_state["relative_quat"],
            dtype=np.float64,
        ).astype(float).tolist(),
        "object_frame_relative_pos": object_frame_relative_pos.astype(float).tolist(),
        "object_frame_relative_pos_norm": float(np.linalg.norm(object_frame_relative_pos)),
        "gripper_qpos": np.asarray(
            current_state["robot0_gripper_qpos"],
            dtype=np.float64,
        ).astype(float).tolist(),
        "eef_object_dist": float(np.linalg.norm(eef_pos - object_pos)),
    }


def mark_square_stage(square_diagnostics, stage, step):
    if square_diagnostics is None or stage not in SQUARE_STAGE_NAMES:
        return
    if not square_diagnostics["stage_success"].get(stage, False):
        square_diagnostics["stage_success"][stage] = True
        square_diagnostics["stage_success_steps"][stage] = int(step)


def square_stage_thresholds(policy):
    distance_config = policy.distance_config
    return {
        "transport_xy": float(
            distance_config.get("square_phase_gate_transport_xy_threshold", 0.16)
        ),
        "pre_insert_xy": float(
            distance_config.get(
                "square_phase_gate_pre_insert_xy_threshold",
                distance_config.get("square_pre_insert_xy_threshold", 0.08),
            )
        ),
        "insert_xy": float(
            distance_config.get(
                "square_phase_gate_insert_xy_threshold",
                distance_config.get("square_insert_xy_threshold", 0.04),
            )
        ),
        "insert_z": float(
            distance_config.get(
                "square_phase_gate_insert_z_threshold",
                distance_config.get("square_insert_z_threshold", 0.94),
            )
        ),
    }


def phase_reached(phase_info, phase_name):
    target_id = SQUARE_PHASE_IDS[phase_name]
    for key in ("phase_id", "raw_phase_id"):
        phase_id = phase_info.get(key)
        if phase_id is not None and int(phase_id) >= target_id:
            return True
    return False


def update_square_stage_diagnostics(
    square_diagnostics,
    phase_info,
    step_diag,
    pick_success,
    task_success,
    step,
):
    if square_diagnostics is None:
        return
    if bool(phase_info.get("lift_start_ready", False)) or bool(pick_success):
        mark_square_stage(square_diagnostics, "pick", step)
    if pick_success:
        mark_square_stage(square_diagnostics, "lift", step)

        thresholds = square_diagnostics["stage_thresholds"]
        xy = float(step_diag["object_to_target_xy"])
        object_z = float(step_diag["object_z"])
        if (
            phase_reached(phase_info, "transport")
            or xy <= float(thresholds["transport_xy"])
        ):
            mark_square_stage(square_diagnostics, "transport", step)
        if (
            phase_reached(phase_info, "pre_insert")
            or xy <= float(thresholds["pre_insert_xy"])
        ):
            mark_square_stage(square_diagnostics, "pre_insert", step)
        if (
            phase_reached(phase_info, "insert")
            or (
                xy <= float(thresholds["insert_xy"])
                and object_z <= float(thresholds["insert_z"])
            )
        ):
            mark_square_stage(square_diagnostics, "insert", step)
    if task_success:
        for stage in SQUARE_STAGE_NAMES:
            mark_square_stage(square_diagnostics, stage, step)


def finalize_square_stage_diagnostics(square_diagnostics):
    if square_diagnostics is None:
        return
    latest_stage = None
    first_failed_stage = None
    stage_success = square_diagnostics["stage_success"]
    for stage in SQUARE_STAGE_NAMES:
        if bool(stage_success.get(stage, False)):
            latest_stage = stage
        elif first_failed_stage is None:
            first_failed_stage = stage
    square_diagnostics["latest_stage"] = latest_stage
    square_diagnostics["first_failed_stage"] = first_failed_stage


def mark_can_stage(can_diagnostics, stage, step):
    if can_diagnostics is None or stage not in CAN_STAGE_NAMES:
        return
    if not can_diagnostics["stage_success"].get(stage, False):
        can_diagnostics["stage_success"][stage] = True
        can_diagnostics["stage_success_steps"][stage] = int(step)


def can_stage_thresholds(policy):
    distance_config = policy.distance_config
    return {
        "transport_xy": float(
            distance_config.get("can_transport_success_xy_threshold", 0.08)
        ),
        "gripper_close": float(
            distance_config.get(
                "can_phase_gate_lift_start_gripper_close_threshold",
                0.03,
            )
        ),
        "lift_delta": float(
            distance_config.get("can_phase_gate_pick_lift_delta", 0.03)
        ),
    }


def can_phase_reached(phase_info, phase_name):
    target_id = CAN_PHASE_IDS[phase_name]
    for key in ("phase_id", "raw_phase_id"):
        phase_id = phase_info.get(key)
        if phase_id is not None and int(phase_id) >= target_id:
            return True
    return False


def update_can_stage_diagnostics(
    can_diagnostics,
    phase_info,
    step_diag,
    pick_info,
    pick_success,
    task_success,
    step,
):
    if can_diagnostics is None:
        return
    thresholds = can_diagnostics["stage_thresholds"]
    gripper = np.asarray(step_diag.get("gripper_qpos", []), dtype=np.float64)
    gripper_closed = bool(
        gripper.size > 0 and float(gripper[0]) <= float(thresholds["gripper_close"])
    )
    if bool(phase_info.get("lift_start_ready", False)) or gripper_closed:
        mark_can_stage(can_diagnostics, "pick", step)
    if pick_success:
        mark_can_stage(can_diagnostics, "pick", step)
        mark_can_stage(can_diagnostics, "lift", step)
        xy = float(step_diag["object_to_target_xy"])
        if (
            can_phase_reached(phase_info, "transport")
            or xy <= float(thresholds["transport_xy"])
        ):
            mark_can_stage(can_diagnostics, "transport", step)
    if task_success:
        for stage in CAN_STAGE_NAMES:
            mark_can_stage(can_diagnostics, stage, step)


def finalize_can_stage_diagnostics(can_diagnostics):
    if can_diagnostics is None:
        return
    latest_stage = None
    first_failed_stage = None
    stage_success = can_diagnostics["stage_success"]
    for stage in CAN_STAGE_NAMES:
        if bool(stage_success.get(stage, False)):
            latest_stage = stage
        elif first_failed_stage is None:
            first_failed_stage = stage
    can_diagnostics["latest_stage"] = latest_stage
    can_diagnostics["first_failed_stage"] = first_failed_stage


def can_stage_success_for_result(result):
    diagnostics = result.get("can_diagnostics") or {}
    return diagnostics.get("stage_success") or {}


def can_stage_steps_for_result(result):
    diagnostics = result.get("can_diagnostics") or {}
    return diagnostics.get("stage_success_steps") or {}


def summarize_can_stages(results):
    if not results:
        return {}, {}, {}

    stage_summary = {}
    total = len(results)
    for stage in CAN_STAGE_NAMES:
        stage_results = [
            result
            for result in results
            if bool(can_stage_success_for_result(result).get(stage, False))
        ]
        step_values = [
            can_stage_steps_for_result(result).get(stage)
            for result in stage_results
            if can_stage_steps_for_result(result).get(stage) is not None
        ]
        stage_summary[stage] = {
            "rate": len(stage_results) / total,
            "num_success": len(stage_results),
            "num_total": total,
            "successful_seeds": [result["seed"] for result in stage_results],
            "failed_seeds": [
                result["seed"]
                for result in results
                if not bool(can_stage_success_for_result(result).get(stage, False))
            ],
            "avg_first_step": float(np.mean(step_values)) if step_values else None,
        }

    transition_summary = {}
    for previous_stage, next_stage in zip(CAN_STAGE_NAMES[:-1], CAN_STAGE_NAMES[1:]):
        denominator = [
            result
            for result in results
            if bool(can_stage_success_for_result(result).get(previous_stage, False))
        ]
        numerator = [
            result
            for result in denominator
            if bool(can_stage_success_for_result(result).get(next_stage, False))
        ]
        transition_summary[f"{previous_stage}_to_{next_stage}"] = {
            "rate": (len(numerator) / len(denominator)) if denominator else None,
            "num_success": len(numerator),
            "num_total": len(denominator),
            "successful_seeds": [result["seed"] for result in numerator],
            "failed_seeds": [
                result["seed"]
                for result in denominator
                if not bool(can_stage_success_for_result(result).get(next_stage, False))
            ],
        }

    first_failed = {
        stage: {"num": 0, "seeds": []}
        for stage in [*CAN_STAGE_NAMES, "none", "unknown"]
    }
    for result in results:
        diagnostics = result.get("can_diagnostics") or {}
        stage = diagnostics.get("first_failed_stage")
        if stage is None:
            stage = "none" if diagnostics.get("stage_success") else "unknown"
        if stage not in first_failed:
            first_failed[stage] = {"num": 0, "seeds": []}
        first_failed[stage]["num"] += 1
        first_failed[stage]["seeds"].append(result["seed"])
    first_failed = {
        stage: payload
        for stage, payload in first_failed.items()
        if payload["num"] > 0
    }
    return stage_summary, transition_summary, first_failed


def square_stage_success_for_result(result):
    diagnostics = result.get("square_diagnostics") or {}
    return diagnostics.get("stage_success") or {}


def square_stage_steps_for_result(result):
    diagnostics = result.get("square_diagnostics") or {}
    return diagnostics.get("stage_success_steps") or {}


def summarize_square_stages(results):
    if not results:
        return {}, {}, {}

    stage_summary = {}
    total = len(results)
    for stage in SQUARE_STAGE_NAMES:
        stage_results = [
            result
            for result in results
            if bool(square_stage_success_for_result(result).get(stage, False))
        ]
        step_values = [
            square_stage_steps_for_result(result).get(stage)
            for result in stage_results
            if square_stage_steps_for_result(result).get(stage) is not None
        ]
        stage_summary[stage] = {
            "rate": len(stage_results) / total,
            "num_success": len(stage_results),
            "num_total": total,
            "successful_seeds": [result["seed"] for result in stage_results],
            "failed_seeds": [
                result["seed"]
                for result in results
                if not bool(square_stage_success_for_result(result).get(stage, False))
            ],
            "avg_first_step": float(np.mean(step_values)) if step_values else None,
        }

    transition_summary = {}
    for previous_stage, next_stage in zip(SQUARE_STAGE_NAMES[:-1], SQUARE_STAGE_NAMES[1:]):
        denominator = [
            result
            for result in results
            if bool(square_stage_success_for_result(result).get(previous_stage, False))
        ]
        numerator = [
            result
            for result in denominator
            if bool(square_stage_success_for_result(result).get(next_stage, False))
        ]
        transition_summary[f"{previous_stage}_to_{next_stage}"] = {
            "rate": (len(numerator) / len(denominator)) if denominator else None,
            "num_success": len(numerator),
            "num_total": len(denominator),
            "successful_seeds": [result["seed"] for result in numerator],
            "failed_seeds": [
                result["seed"]
                for result in denominator
                if not bool(square_stage_success_for_result(result).get(next_stage, False))
            ],
        }

    first_failed = {
        stage: {"num": 0, "seeds": []}
        for stage in [*SQUARE_STAGE_NAMES, "none", "unknown"]
    }
    for result in results:
        diagnostics = result.get("square_diagnostics") or {}
        stage = diagnostics.get("first_failed_stage")
        if stage is None:
            stage = "none" if diagnostics.get("stage_success") else "unknown"
        if stage not in first_failed:
            first_failed[stage] = {"num": 0, "seeds": []}
        first_failed[stage]["num"] += 1
        first_failed[stage]["seeds"].append(result["seed"])
    first_failed = {
        stage: payload
        for stage, payload in first_failed.items()
        if payload["num"] > 0
    }
    return stage_summary, transition_summary, first_failed
