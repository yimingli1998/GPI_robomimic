import time
from copy import deepcopy

import h5py
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from .config import CAN_STAGE_NAMES, RAW_OBS_KEYS, SQUARE_STAGE_NAMES
from .diagnostics import (
    can_stage_thresholds,
    finalize_can_stage_diagnostics,
    finalize_square_stage_diagnostics,
    pick_success_info,
    square_stage_thresholds,
    square_step_diagnostics,
    update_can_stage_diagnostics,
    update_square_stage_diagnostics,
)
from .state import (
    maybe_unwrap_basic_for_robosuite_14,
    patch_egl_probe,
    set_abs_controller,
    set_delta_controller,
)


def create_env(dataset_path, control_mode="abs", render_offscreen=False):
    if render_offscreen:
        patch_egl_probe()
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": RAW_OBS_KEYS})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    if control_mode == "abs":
        set_abs_controller(env_meta)
    elif control_mode == "delta":
        set_delta_controller(env_meta)
    else:
        raise ValueError(f"Unknown control mode: {control_mode}")
    maybe_unwrap_basic_for_robosuite_14(env_meta)
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=bool(render_offscreen),
        use_image_obs=False,
    )


def validate_alignment(dataset_path, policy, env, control_mode):
    with h5py.File(dataset_path, "r") as f:
        demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
        total_actions = sum(len(f[f"data/{key}/actions"]) for key in demo_keys)
        obs_dim = sum(f[f"data/{demo_keys[0]}/obs/{key}"].shape[1] for key in RAW_OBS_KEYS)
        action_dim = f[f"data/{demo_keys[0]}/actions"].shape[1]
    env_action_dim = getattr(env.env, "action_dim", getattr(env, "action_dimension", None))
    print(
        "Alignment check: "
        f"demos={len(demo_keys)}, pairs={total_actions}, raw_obs_dim={obs_dim}, "
        f"metric_dim={policy.state_tensor.shape[1]}, dataset_action_dim={action_dim}, "
        f"env_action_dim={env_action_dim}, env_control_mode={control_mode}, "
        f"demo_object_pose_start={policy.demo_object_pose_start}, "
        f"live_object_pose_start={policy.object_pose_start}"
    )


def render_video_frame(env, camera_names, height=512, width=512):
    frames = [
        env.render(
            mode="rgb_array",
            height=int(height),
            width=int(width),
            camera_name=camera_name,
        )
        for camera_name in camera_names
    ]
    return np.concatenate(frames, axis=1)


def rollout(
    policy,
    env,
    seed,
    horizon,
    set_seed=True,
    success_mode="task",
    pick_success_lift_delta=0.03,
    pick_success_eef_object_max=0.12,
    pick_success_hold_steps=1,
    pick_success_baseline="table_min",
    pick_success_demo_calibrated=True,
    video_writer=None,
    video_skip=5,
    video_camera_names=None,
    video_height=512,
    video_width=512,
):
    if set_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)
    initial_object_z = float(
        np.asarray(obs["object"])[policy.object_pose_start + 2]
    )
    table_object_z = initial_object_z
    total_reward = 0.0
    success = False
    task_success = False
    pick_hold_steps = 0
    pick_success = False
    pick_success_step = None
    last_pick_info = None
    effective_pick_success_lift_delta = float(pick_success_lift_delta)
    effective_pick_success_eef_object_max = float(pick_success_eef_object_max)
    if bool(pick_success_demo_calibrated):
        demo_thresholds = getattr(policy, "demo_lift_success_thresholds", {})
        if demo_thresholds:
            effective_pick_success_lift_delta = float(
                demo_thresholds.get(
                    "lift_delta",
                    effective_pick_success_lift_delta,
                )
            )
            effective_pick_success_eef_object_max = float(
                demo_thresholds.get(
                    "eef_object_max",
                    effective_pick_success_eef_object_max,
                )
            )
    square_diagnostics = None
    can_diagnostics = None
    if policy.task == "square":
        square_diagnostics = {
            "final": None,
            "first_phase_step": {},
            "stage_thresholds": square_stage_thresholds(policy),
            "stage_success": {stage: False for stage in SQUARE_STAGE_NAMES},
            "stage_success_steps": {stage: None for stage in SQUARE_STAGE_NAMES},
            "latest_stage": None,
            "first_failed_stage": None,
            "min_object_to_target_xy_after_pick": None,
            "min_object_to_target_xy_after_pick_step": None,
            "min_object_to_target_xy_after_pick_diag": None,
            "min_object_z_after_pick": None,
            "min_object_z_after_pick_step": None,
            "min_object_z_when_pre_insert_xy_after_pick": None,
            "min_object_z_when_pre_insert_xy_after_pick_step": None,
            "min_object_z_when_insert_xy_after_pick": None,
            "min_object_z_when_insert_xy_after_pick_step": None,
            "max_object_lift_after_pick": None,
            "max_eef_object_dist_after_pick": None,
            "max_eef_object_dist_after_pick_step": None,
            "final_pick_success_still_true": None,
            "dropped_or_lost_after_pick": False,
            "reached_pre_insert_xy_after_pick": False,
            "reached_insert_xy_after_pick": False,
        }
    elif policy.task == "can":
        can_diagnostics = {
            "final": None,
            "first_phase_step": {},
            "stage_thresholds": can_stage_thresholds(policy),
            "stage_success": {stage: False for stage in CAN_STAGE_NAMES},
            "stage_success_steps": {stage: None for stage in CAN_STAGE_NAMES},
            "latest_stage": None,
            "first_failed_stage": None,
            "min_object_to_target_xy_after_pick": None,
            "min_object_to_target_xy_after_pick_step": None,
            "max_object_lift_after_pick": None,
            "final_pick_success_still_true": None,
            "dropped_or_lost_after_pick": False,
        }
    video_count = 0
    if video_camera_names is None:
        video_camera_names = ["agentview"]
    start_time = time.time()
    for step_i in range(horizon):
        env_state = None
        action = policy(obs, env_state)
        obs, reward, done, _ = env.step(action)
        current_object_z = float(
            np.asarray(obs["object"])[policy.object_pose_start + 2]
        )
        table_object_z = min(float(table_object_z), current_object_z)
        total_reward += float(reward)
        task_success = bool(env.is_success()["task"])
        last_pick_info = pick_success_info(
            obs,
            policy.object_pose_start,
            initial_object_z,
            table_object_z,
            effective_pick_success_lift_delta,
            effective_pick_success_eef_object_max,
            baseline=pick_success_baseline,
        )
        if last_pick_info["success"]:
            pick_hold_steps += 1
        else:
            pick_hold_steps = 0
        if (
            not pick_success
            and pick_hold_steps >= max(1, int(pick_success_hold_steps))
        ):
            pick_success = True
            pick_success_step = step_i + 1
        if square_diagnostics is not None:
            step_diag = square_step_diagnostics(obs, policy)
            phase_info = getattr(policy, "_last_square_phase_info", {}) or {}
            phase = phase_info.get("phase")
            if phase is not None:
                square_diagnostics["first_phase_step"].setdefault(
                    str(phase),
                    int(step_i + 1),
                )
            square_diagnostics["final"] = {
                **step_diag,
                "phase": None if phase is None else str(phase),
                "step": int(step_i + 1),
            }
            update_square_stage_diagnostics(
                square_diagnostics,
                phase_info,
                step_diag,
                pick_success,
                task_success,
                step_i + 1,
            )
            if pick_success:
                xy = float(step_diag["object_to_target_xy"])
                best_xy = square_diagnostics["min_object_to_target_xy_after_pick"]
                if best_xy is None or xy < float(best_xy):
                    square_diagnostics["min_object_to_target_xy_after_pick"] = xy
                    square_diagnostics["min_object_to_target_xy_after_pick_step"] = (
                        int(step_i + 1)
                    )
                    square_diagnostics["min_object_to_target_xy_after_pick_diag"] = {
                        **step_diag,
                        "phase": None if phase is None else str(phase),
                        "step": int(step_i + 1),
                    }
                object_z = float(step_diag["object_z"])
                best_z = square_diagnostics["min_object_z_after_pick"]
                if best_z is None or object_z < float(best_z):
                    square_diagnostics["min_object_z_after_pick"] = object_z
                    square_diagnostics["min_object_z_after_pick_step"] = int(step_i + 1)
                lift = float(last_pick_info["object_lift"])
                max_lift = square_diagnostics["max_object_lift_after_pick"]
                if max_lift is None or lift > float(max_lift):
                    square_diagnostics["max_object_lift_after_pick"] = lift
                eef_object_dist = float(step_diag["eef_object_dist"])
                max_dist = square_diagnostics["max_eef_object_dist_after_pick"]
                if max_dist is None or eef_object_dist > float(max_dist):
                    square_diagnostics["max_eef_object_dist_after_pick"] = (
                        eef_object_dist
                    )
                    square_diagnostics["max_eef_object_dist_after_pick_step"] = int(
                        step_i + 1
                    )
                thresholds = square_diagnostics["stage_thresholds"]
                pre_insert_xy = float(thresholds["pre_insert_xy"])
                insert_xy = float(thresholds["insert_xy"])
                square_diagnostics["reached_pre_insert_xy_after_pick"] = bool(
                    square_diagnostics["reached_pre_insert_xy_after_pick"]
                    or xy <= pre_insert_xy
                )
                square_diagnostics["reached_insert_xy_after_pick"] = bool(
                    square_diagnostics["reached_insert_xy_after_pick"]
                    or xy <= insert_xy
                )
                if xy <= pre_insert_xy:
                    best_pre_z = square_diagnostics[
                        "min_object_z_when_pre_insert_xy_after_pick"
                    ]
                    if best_pre_z is None or object_z < float(best_pre_z):
                        square_diagnostics[
                            "min_object_z_when_pre_insert_xy_after_pick"
                        ] = object_z
                        square_diagnostics[
                            "min_object_z_when_pre_insert_xy_after_pick_step"
                        ] = int(step_i + 1)
                if xy <= insert_xy:
                    best_insert_z = square_diagnostics[
                        "min_object_z_when_insert_xy_after_pick"
                    ]
                    if best_insert_z is None or object_z < float(best_insert_z):
                        square_diagnostics[
                            "min_object_z_when_insert_xy_after_pick"
                        ] = object_z
                        square_diagnostics[
                            "min_object_z_when_insert_xy_after_pick_step"
                        ] = int(step_i + 1)
        if can_diagnostics is not None:
            step_diag = square_step_diagnostics(obs, policy)
            phase_info = getattr(policy, "_last_can_phase_info", {}) or {}
            phase = phase_info.get("phase")
            if phase is not None:
                can_diagnostics["first_phase_step"].setdefault(
                    str(phase),
                    int(step_i + 1),
                )
            can_diagnostics["final"] = {
                **step_diag,
                "phase": None if phase is None else str(phase),
                "step": int(step_i + 1),
            }
            update_can_stage_diagnostics(
                can_diagnostics,
                phase_info,
                step_diag,
                last_pick_info,
                pick_success,
                task_success,
                step_i + 1,
            )
            if pick_success:
                xy = float(step_diag["object_to_target_xy"])
                best_xy = can_diagnostics["min_object_to_target_xy_after_pick"]
                if best_xy is None or xy < float(best_xy):
                    can_diagnostics["min_object_to_target_xy_after_pick"] = xy
                    can_diagnostics["min_object_to_target_xy_after_pick_step"] = (
                        int(step_i + 1)
                    )
                lift = float(last_pick_info["object_lift"])
                max_lift = can_diagnostics["max_object_lift_after_pick"]
                if max_lift is None or lift > float(max_lift):
                    can_diagnostics["max_object_lift_after_pick"] = lift
        if video_writer is not None:
            if video_count % max(1, int(video_skip)) == 0:
                video_writer.append_data(
                    render_video_frame(
                        env,
                        video_camera_names,
                        height=video_height,
                        width=video_width,
                    )
                )
            video_count += 1
        if success_mode == "pick":
            success = pick_success
        elif success_mode == "task":
            success = task_success
        else:
            raise ValueError(f"Unknown success_mode: {success_mode}")
        if success or done:
            break
        obs = deepcopy(obs)
    if square_diagnostics is not None:
        square_diagnostics["final_pick_success_still_true"] = (
            None if last_pick_info is None else bool(last_pick_info["success"])
        )
        square_diagnostics["dropped_or_lost_after_pick"] = bool(
            pick_success
            and last_pick_info is not None
            and not bool(last_pick_info["success"])
        )
        finalize_square_stage_diagnostics(square_diagnostics)
    if can_diagnostics is not None:
        can_diagnostics["final_pick_success_still_true"] = (
            None if last_pick_info is None else bool(last_pick_info["success"])
        )
        can_diagnostics["dropped_or_lost_after_pick"] = bool(
            pick_success
            and last_pick_info is not None
            and not bool(last_pick_info["success"])
        )
        finalize_can_stage_diagnostics(can_diagnostics)
    return {
        "seed": int(seed),
        "success": bool(success),
        "success_mode": str(success_mode),
        "task_success": bool(task_success),
        "pick_success": bool(pick_success),
        "pick_success_step": (
            None if pick_success_step is None else int(pick_success_step)
        ),
        "pick_success_info": last_pick_info,
        "pick_success_thresholds": {
            "lift_delta": float(effective_pick_success_lift_delta),
            "eef_object_max": float(effective_pick_success_eef_object_max),
            "demo_calibrated": bool(pick_success_demo_calibrated),
        },
        "return": float(total_reward),
        "horizon": int(step_i + 1),
        "time": float(time.time() - start_time),
        "initial_state_info": getattr(policy, "_initial_state_info", {}),
        "metric_phase_counts": getattr(policy, "_metric_phase_counts", {}),
        "phase_gate_counts": getattr(policy, "_phase_gate_counts", {}),
        "lift_start_preference_counts": getattr(policy, "_lift_start_preference_counts", {}),
        "square_diagnostics": square_diagnostics,
        "can_diagnostics": can_diagnostics,
        "last_square_phase_info": getattr(policy, "_last_square_phase_info", {}),
        "last_can_phase_info": getattr(policy, "_last_can_phase_info", {}),
        "last_action_info": getattr(policy, "_last_action_info", {}),
    }
