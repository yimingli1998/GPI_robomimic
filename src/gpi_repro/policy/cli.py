import argparse
import json
import pathlib

import imageio
import numpy as np
import torch

from .config import CAN_STAGE_NAMES, SQUARE_PHASE_NAMES, SQUARE_STAGE_NAMES, parse_seeds
from .diagnostics import summarize_can_stages, summarize_square_stages
from .gpi_policy import AbsAlignedKNNPolicy
from .rollout import create_env, rollout, validate_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["can", "square", "lift"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0..20")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--success-mode",
        choices=["task", "pick"],
        default="task",
        help="Evaluate full task success or stop when the object has been lifted by a stable pick.",
    )
    parser.add_argument("--pick-success-lift-delta", type=float, default=0.03)
    parser.add_argument("--pick-success-eef-object-max", type=float, default=0.12)
    parser.add_argument("--pick-success-hold-steps", type=int, default=1)
    parser.add_argument(
        "--pick-success-baseline",
        choices=["table_min", "initial"],
        default="table_min",
        help="Lift threshold baseline for pick-only evaluation.",
    )
    parser.add_argument(
        "--pick-success-demo-calibrated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the tightest demo-calibrated lift success thresholds for square pick-only evaluation.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional directory to save one rollout mp4 per seed.",
    )
    parser.add_argument(
        "--video-skip",
        type=int,
        default=5,
        help="Write one video frame every N environment steps.",
    )
    parser.add_argument(
        "--video-camera-names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="Camera name(s) to render; multiple cameras are concatenated horizontally.",
    )
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument(
        "--env-control-mode",
        choices=["abs", "delta"],
        default="abs",
        help="Use absolute-control evaluation for DP-style abs actions, or delta-control evaluation for legacy v15 actions.",
    )
    parser.add_argument(
        "--distance-config",
        default=None,
        help="Path to JSON or inline JSON with feature/group/phase distance weights.",
    )
    parser.add_argument(
        "--flow-field",
        action="store_true",
        help="Use a multi-neighbor flow correction instead of executing a queued hard KNN plan.",
    )
    parser.add_argument("--flow-neighbors", type=int, default=8)
    parser.add_argument(
        "--flow-attraction",
        type=float,
        default=1.0,
        help="At 1.0 no extra attraction is added; values above 1 add flow correction.",
    )
    parser.add_argument(
        "--flow-target-offset",
        type=int,
        default=0,
        help="Use each top-N neighbor's future demo state as the flow target.",
    )
    parser.add_argument(
        "--progress-guard",
        action="store_true",
        help="Prevent reselecting earlier states from the current demo segment.",
    )
    parser.add_argument(
        "--progress-guard-min-position",
        type=int,
        default=0,
        help="Only apply same-demo progress guard after this demo timestep.",
    )
    parser.add_argument(
        "--progress-guard-closed-only",
        action="store_true",
        help="Only apply progress guard after the gripper closes.",
    )
    parser.add_argument(
        "--future-progress-mode",
        choices=["off", "candidate"],
        default="off",
        help="Rerank top KNN candidates by demo-derived future progress.",
    )
    parser.add_argument(
        "--future-progress-min-phase",
        choices=SQUARE_PHASE_NAMES,
        default="transport",
        help="First square phase where future-progress reranking is active.",
    )
    parser.add_argument("--future-progress-top-n", type=int, default=32)
    parser.add_argument("--future-progress-horizon", type=int, default=32)
    parser.add_argument("--future-progress-distance-weight", type=float, default=1.0)
    parser.add_argument("--future-progress-xy-weight", type=float, default=1.0)
    parser.add_argument("--future-progress-z-weight", type=float, default=0.5)
    parser.add_argument("--future-progress-ori-weight", type=float, default=0.25)
    parser.add_argument("--future-progress-success-weight", type=float, default=0.25)
    parser.add_argument("--future-progress-success-temp", type=float, default=80.0)
    parser.add_argument("--future-progress-distance-ratio", type=float, default=2.0)
    parser.add_argument("--future-progress-distance-margin", type=float, default=0.05)
    parser.add_argument(
        "--local-residual",
        action="store_true",
        help="Apply analytic eef-from-nut future-pose correction in late square phases.",
    )
    parser.add_argument(
        "--local-residual-min-phase",
        choices=SQUARE_PHASE_NAMES,
        default="pre_insert",
    )
    parser.add_argument("--local-residual-target-offset", type=int, default=4)
    parser.add_argument("--local-residual-pos-alpha", type=float, default=0.35)
    parser.add_argument("--local-residual-ori-alpha", type=float, default=0.20)
    parser.add_argument("--local-residual-max-pos", type=float, default=0.012)
    parser.add_argument("--local-residual-max-rot", type=float, default=0.08)
    parser.add_argument("--pca-state-components", type=int, default=16)
    parser.add_argument(
        "--pca-state-variance-power",
        type=float,
        default=None,
        help="PCA variance scaling exponent: 0 keeps raw PCA scores, 1 fully whitens, values in between partially whiten.",
    )
    parser.add_argument(
        "--pca-state-cache",
        default=None,
        help="Path to a saved PCA state cache with components and demo embeddings.",
    )
    parser.add_argument(
        "--pca-state-rebuild-cache",
        action="store_true",
        help="Refit and overwrite --pca-state-cache even if it already exists.",
    )
    parser.add_argument(
        "--build-pca-state-cache",
        action="store_true",
        help="Build --pca-state-cache from the dataset and exit before creating the rollout environment.",
    )
    parser.add_argument(
        "--fresh-env-per-seed",
        action="store_true",
        help="Create a fresh environment for each seed.",
    )
    parser.add_argument(
        "--action-retarget",
        choices=[
            "none",
            "default",
            "eef_pos_delta",
            "object_abs_delta_xyzw",
            "object_frame_delta_xyzw",
            "phase_eefxyzw_object_abs",
        ],
        default="default",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = None
    write_video = args.video_dir is not None
    if write_video:
        video_dir = pathlib.Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.build_pca_state_cache:
        args.pca_state_rebuild_cache = True
        if args.pca_state_cache is None:
            raise ValueError("--build-pca-state-cache requires --pca-state-cache")
    action_retarget = args.action_retarget
    if args.env_control_mode == "delta" and action_retarget == "default":
        action_retarget = "none"
    policy = AbsAlignedKNNPolicy(
        dataset_path=args.dataset,
        task=args.task,
        k=args.k,
        action_horizon=args.action_horizon,
        action_retarget=action_retarget,
        distance_config=args.distance_config,
        flow_field=args.flow_field,
        flow_neighbors=args.flow_neighbors,
        flow_attraction=args.flow_attraction,
        flow_target_offset=args.flow_target_offset,
        progress_guard=args.progress_guard,
        progress_guard_min_position=args.progress_guard_min_position,
        progress_guard_closed_only=args.progress_guard_closed_only,
        future_progress_mode=args.future_progress_mode,
        future_progress_min_phase=args.future_progress_min_phase,
        future_progress_top_n=args.future_progress_top_n,
        future_progress_horizon=args.future_progress_horizon,
        future_progress_distance_weight=args.future_progress_distance_weight,
        future_progress_xy_weight=args.future_progress_xy_weight,
        future_progress_z_weight=args.future_progress_z_weight,
        future_progress_ori_weight=args.future_progress_ori_weight,
        future_progress_success_weight=args.future_progress_success_weight,
        future_progress_success_temp=args.future_progress_success_temp,
        future_progress_distance_ratio=args.future_progress_distance_ratio,
        future_progress_distance_margin=args.future_progress_distance_margin,
        local_residual=args.local_residual,
        local_residual_min_phase=args.local_residual_min_phase,
        local_residual_target_offset=args.local_residual_target_offset,
        local_residual_pos_alpha=args.local_residual_pos_alpha,
        local_residual_ori_alpha=args.local_residual_ori_alpha,
        local_residual_max_pos=args.local_residual_max_pos,
        local_residual_max_rot=args.local_residual_max_rot,
        pca_state_components=args.pca_state_components,
        pca_state_variance_power=args.pca_state_variance_power,
        pca_state_cache=args.pca_state_cache,
        pca_state_rebuild_cache=args.pca_state_rebuild_cache,
        device=args.device,
    )
    if args.build_pca_state_cache:
        print(f"Built PCA state cache and exiting: {args.pca_state_cache}")
        return
    env = create_env(
        args.dataset,
        args.env_control_mode,
        render_offscreen=write_video,
    )
    validate_alignment(args.dataset, policy, env, args.env_control_mode)

    seeds = parse_seeds(args.seeds)
    results = []
    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{len(seeds)}] seed {seed}...", end=" ", flush=True)
        if args.fresh_env_per_seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            rollout_env = create_env(
                args.dataset,
                args.env_control_mode,
                render_offscreen=write_video,
            )
        else:
            rollout_env = env
        video_path = None
        video_writer = None
        if write_video:
            video_path = video_dir / f"seed_{int(seed):03d}.mp4"
            video_writer = imageio.get_writer(str(video_path), fps=20)
        try:
            result = rollout(
                policy,
                rollout_env,
                seed,
                args.horizon,
                set_seed=not args.fresh_env_per_seed,
                success_mode=args.success_mode,
                pick_success_lift_delta=args.pick_success_lift_delta,
                pick_success_eef_object_max=args.pick_success_eef_object_max,
                pick_success_hold_steps=args.pick_success_hold_steps,
                pick_success_baseline=args.pick_success_baseline,
                pick_success_demo_calibrated=args.pick_success_demo_calibrated,
                video_writer=video_writer,
                video_skip=args.video_skip,
                video_camera_names=args.video_camera_names,
                video_height=args.video_height,
                video_width=args.video_width,
            )
        finally:
            if video_writer is not None:
                video_writer.close()
        if video_path is not None:
            result["video_path"] = str(video_path)
        results.append(result)
        status = "success" if result["success"] else "fail"
        pick_status = "pick" if result.get("pick_success") else "no-pick"
        square_diag = result.get("square_diagnostics") or {}
        can_diag = result.get("can_diagnostics") or {}
        latest_stage = square_diag.get("latest_stage") or can_diag.get("latest_stage")
        stage_status = "" if latest_stage is None else f" stage={latest_stage}"
        print(
            f"{status} {pick_status} steps={result['horizon']} "
            f"time={result['time']:.2f}s{stage_status}"
        )

    successes = [result for result in results if result["success"]]
    failures = [result for result in results if not result["success"]]
    pick_successes = [result for result in results if result.get("pick_success")]
    pick_failures = [result for result in results if not result.get("pick_success")]
    if args.task == "square":
        (
            square_stage_summary,
            square_stage_transition_summary,
            square_first_failed_stage,
        ) = summarize_square_stages(results)
        can_stage_summary = {}
        can_stage_transition_summary = {}
        can_first_failed_stage = {}
    elif args.task == "can":
        square_stage_summary = {}
        square_stage_transition_summary = {}
        square_first_failed_stage = {}
        (
            can_stage_summary,
            can_stage_transition_summary,
            can_first_failed_stage,
        ) = summarize_can_stages(results)
    else:
        square_stage_summary = {}
        square_stage_transition_summary = {}
        square_first_failed_stage = {}
        can_stage_summary = {}
        can_stage_transition_summary = {}
        can_first_failed_stage = {}
    summary = {
        "dataset": args.dataset,
        "task": args.task,
        "seeds": seeds,
        "success_rate": len(successes) / len(results),
        "num_success": len(successes),
        "num_total": len(results),
        "successful_seeds": [result["seed"] for result in successes],
        "failed_seeds": [result["seed"] for result in failures],
        "pick_success_rate": len(pick_successes) / len(results),
        "num_pick_success": len(pick_successes),
        "pick_successful_seeds": [result["seed"] for result in pick_successes],
        "pick_failed_seeds": [result["seed"] for result in pick_failures],
        "square_stage_success": square_stage_summary,
        "square_stage_transitions": square_stage_transition_summary,
        "square_first_failed_stage": square_first_failed_stage,
        "can_stage_success": can_stage_summary,
        "can_stage_transitions": can_stage_transition_summary,
        "can_first_failed_stage": can_first_failed_stage,
        "avg_steps_success": float(np.mean([result["horizon"] for result in successes])) if successes else 0.0,
        "avg_steps_pick_success": (
            float(np.mean([
                result["pick_success_step"]
                for result in pick_successes
                if result.get("pick_success_step") is not None
            ]))
            if pick_successes
            else 0.0
        ),
        "avg_time": float(np.mean([result["time"] for result in results])) if results else 0.0,
        "config": {
            "k": args.k,
            "action_horizon": policy.action_horizon,
            "future_progress_mode": args.future_progress_mode,
            "future_progress_min_phase": args.future_progress_min_phase,
            "future_progress_top_n": args.future_progress_top_n,
            "future_progress_horizon": args.future_progress_horizon,
            "future_progress_distance_weight": args.future_progress_distance_weight,
            "future_progress_xy_weight": args.future_progress_xy_weight,
            "future_progress_z_weight": args.future_progress_z_weight,
            "future_progress_ori_weight": args.future_progress_ori_weight,
            "future_progress_success_weight": args.future_progress_success_weight,
            "future_progress_success_temp": args.future_progress_success_temp,
            "future_progress_distance_ratio": args.future_progress_distance_ratio,
            "future_progress_distance_margin": args.future_progress_distance_margin,
            "future_progress_selected_count": getattr(
                policy,
                "_future_progress_selected_count",
                0,
            ),
            "local_residual": args.local_residual,
            "local_residual_min_phase": args.local_residual_min_phase,
            "local_residual_target_offset": args.local_residual_target_offset,
            "local_residual_pos_alpha": args.local_residual_pos_alpha,
            "local_residual_ori_alpha": args.local_residual_ori_alpha,
            "local_residual_max_pos": args.local_residual_max_pos,
            "local_residual_max_rot": args.local_residual_max_rot,
            "local_residual_count": getattr(policy, "_local_residual_count", 0),
            "horizon": args.horizon,
            "success_mode": args.success_mode,
            "pick_success_lift_delta": args.pick_success_lift_delta,
            "pick_success_eef_object_max": args.pick_success_eef_object_max,
            "pick_success_hold_steps": args.pick_success_hold_steps,
            "pick_success_baseline": args.pick_success_baseline,
            "pick_success_demo_calibrated": args.pick_success_demo_calibrated,
            "video_dir": None if video_dir is None else str(video_dir),
            "video_skip": args.video_skip,
            "video_camera_names": args.video_camera_names,
            "video_height": args.video_height,
            "video_width": args.video_width,
            "effective_pick_success_thresholds": (
                successes[0].get("pick_success_thresholds")
                if successes
                else (
                    results[0].get("pick_success_thresholds")
                    if results
                    else None
                )
            ),
            "env_control_mode": args.env_control_mode,
            "knn_space": "pca_state",
            "pca_state_components": args.pca_state_components,
            "pca_state_variance_power": args.pca_state_variance_power,
            "pca_state_cache": args.pca_state_cache,
            "pca_state_rebuild_cache": args.pca_state_rebuild_cache,
            "distance_metric": "pca_state_metric",
            "normalizer": "current_mean_std_per_metric_component",
            "output_action": (
                "delta_osc_pose_7d_from_dataset"
                if args.env_control_mode == "delta"
                else "absolute_osc_pose_7d_from_low_dim_abs"
            ),
            "action_retarget": action_retarget,
            "live_object_pose_start": policy.object_pose_start,
            "distance_config": policy.distance_config,
            "demo_lift_calibration": getattr(policy, "demo_lift_calibration", {}),
            "demo_stage_calibration": getattr(policy, "demo_stage_calibration", {}),
            "demo_phase_counts": getattr(policy, "demo_phase_counts", {}),
            "demo_lift_start_window_count": getattr(
                policy,
                "demo_lift_start_window_count",
                0,
            ),
            "demo_lift_start_profile": {
                key: np.asarray(value).astype(float).tolist()
                for key, value in getattr(
                    policy,
                    "demo_lift_start_profile",
                    {},
                ).items()
            },
            "flow_field": args.flow_field,
            "flow_neighbors": args.flow_neighbors,
            "flow_attraction": args.flow_attraction,
            "flow_target_offset": args.flow_target_offset,
            "progress_guard": args.progress_guard,
            "progress_guard_min_position": args.progress_guard_min_position,
            "progress_guard_closed_only": args.progress_guard_closed_only,
            "live_object_pose_start": policy.object_pose_start,
            "fresh_env_per_seed": args.fresh_env_per_seed,
        },
        "results": results,
    }
    out_path = output_dir / "eval_log.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print("=" * 60)
    print(f"Success rate: {summary['success_rate'] * 100:.1f}% ({len(successes)}/{len(results)})")
    print(
        f"Pick success rate: {summary['pick_success_rate'] * 100:.1f}% "
        f"({len(pick_successes)}/{len(results)})"
    )
    if square_stage_summary:
        print("Square stage success rates:")
        for stage in SQUARE_STAGE_NAMES:
            stage_info = square_stage_summary[stage]
            print(
                f"  {stage}: {stage_info['rate'] * 100:.1f}% "
                f"({stage_info['num_success']}/{stage_info['num_total']})"
            )
        print("Square stage transition rates:")
        for name, transition_info in square_stage_transition_summary.items():
            rate = transition_info["rate"]
            rate_text = "n/a" if rate is None else f"{rate * 100:.1f}%"
            print(
                f"  {name}: {rate_text} "
                f"({transition_info['num_success']}/{transition_info['num_total']})"
            )
        print(f"Square first failed stage: {square_first_failed_stage}")
    print(f"Successful seeds: {summary['successful_seeds']}")
    print(f"Failed seeds: {summary['failed_seeds']}")
    print(f"Pick failed seeds: {summary['pick_failed_seeds']}")
    print(f"Saved: {out_path}")
