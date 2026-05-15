import pathlib

import h5py
import numpy as np
import torch

from .action import ActionMixin
from .config import *
from .pca import PCAMixin
from .phase import PhaseMixin
from .state import live_object_pose_start, state_from_demo_obs


class AbsAlignedKNNPolicy(PCAMixin, PhaseMixin, ActionMixin):
    def __init__(
        self,
        dataset_path,
        task,
        k=1,
        action_horizon=None,
        action_retarget="default",
        flow_field=False,
        flow_neighbors=8,
        flow_attraction=1.0,
        flow_target_offset=0,
        progress_guard=False,
        progress_guard_min_position=0,
        progress_guard_closed_only=False,
        future_progress_mode="off",
        future_progress_min_phase="transport",
        future_progress_top_n=32,
        future_progress_horizon=32,
        future_progress_distance_weight=1.0,
        future_progress_xy_weight=1.0,
        future_progress_z_weight=0.5,
        future_progress_ori_weight=0.25,
        future_progress_success_weight=0.25,
        future_progress_success_temp=80.0,
        future_progress_distance_ratio=2.0,
        future_progress_distance_margin=0.05,
        local_residual=False,
        local_residual_min_phase="pre_insert",
        local_residual_target_offset=4,
        local_residual_pos_alpha=0.35,
        local_residual_ori_alpha=0.20,
        local_residual_max_pos=0.012,
        local_residual_max_rot=0.08,
        pca_state_components=16,
        pca_state_variance_power=None,
        pca_state_cache=None,
        pca_state_rebuild_cache=False,
        distance_config=None,
        device="auto",
    ):
        self.dataset_path = dataset_path
        self.task = task
        self.task_profile = TASK_GPI_PROFILES.get(task)
        self.k = int(k)
        if action_horizon is None:
            action_horizon = 8 if task == "square" else 3
        self.action_horizon = max(1, int(action_horizon))
        self.future_progress_mode = str(future_progress_mode)
        if self.future_progress_mode not in {"off", "candidate"}:
            raise ValueError(f"Unknown future_progress_mode: {self.future_progress_mode}")
        self.future_progress_min_phase = str(future_progress_min_phase)
        if self.future_progress_min_phase not in SQUARE_PHASE_IDS:
            raise ValueError(
                f"Unknown future_progress_min_phase: {self.future_progress_min_phase}"
            )
        self.future_progress_min_phase_id = int(
            SQUARE_PHASE_IDS[self.future_progress_min_phase]
        )
        self.future_progress_top_n = max(1, int(future_progress_top_n))
        self.future_progress_horizon = max(1, int(future_progress_horizon))
        self.future_progress_distance_weight = float(future_progress_distance_weight)
        self.future_progress_xy_weight = float(future_progress_xy_weight)
        self.future_progress_z_weight = float(future_progress_z_weight)
        self.future_progress_ori_weight = float(future_progress_ori_weight)
        self.future_progress_success_weight = float(future_progress_success_weight)
        self.future_progress_success_temp = max(1e-6, float(future_progress_success_temp))
        self.future_progress_distance_ratio = float(future_progress_distance_ratio)
        self.future_progress_distance_margin = float(future_progress_distance_margin)
        self.local_residual = bool(local_residual)
        self.local_residual_min_phase = str(local_residual_min_phase)
        if self.local_residual_min_phase not in SQUARE_PHASE_IDS:
            raise ValueError(f"Unknown local_residual_min_phase: {self.local_residual_min_phase}")
        self.local_residual_min_phase_id = int(SQUARE_PHASE_IDS[self.local_residual_min_phase])
        self.local_residual_target_offset = max(1, int(local_residual_target_offset))
        self.local_residual_pos_alpha = float(np.clip(float(local_residual_pos_alpha), 0.0, 1.0))
        self.local_residual_ori_alpha = float(np.clip(float(local_residual_ori_alpha), 0.0, 1.0))
        self.local_residual_max_pos = max(0.0, float(local_residual_max_pos))
        self.local_residual_max_rot = max(0.0, float(local_residual_max_rot))
        self.vote_top_n = 32
        self.vote_window = 16
        self.vote_min_support = 3
        self.vote_support_margin = 2
        self.vote_score_ratio = 1.5
        self.vote_distance_ratio = 1.5
        self.vote_distance_margin = 0.02
        self.demo_object_pose_start = 0
        if action_retarget == "default":
            action_retarget = (
                "phase_eefxyzw_object_abs" if task == "square" else "eef_pos_delta"
            )
        self.action_retarget = action_retarget
        self.flow_field = bool(flow_field)
        self.flow_neighbors = max(1, int(flow_neighbors))
        self.flow_attraction = float(flow_attraction)
        self.flow_attraction_clip = 0.05
        self.flow_target_offset = max(0, int(flow_target_offset))
        self.progress_guard = bool(progress_guard)
        self.progress_backtrack_window = 0
        self.progress_guard_min_position = max(0, int(progress_guard_min_position))
        self.progress_guard_closed_only = bool(progress_guard_closed_only)
        self.knn_space = "pca_state"
        self.pca_state_components = int(pca_state_components)
        if pca_state_variance_power is None:
            pca_state_variance_power = 0.0
        self.pca_state_variance_power = float(pca_state_variance_power)
        self.pca_state_cache = (
            None if pca_state_cache is None else pathlib.Path(pca_state_cache)
        )
        self.pca_state_rebuild_cache = bool(pca_state_rebuild_cache)
        self.pca_state_mean = None
        self.pca_state_components_matrix = None
        self.pca_state_explained_variance = None
        self.pca_state_demo_tensor = None
        self.object_pose_start = live_object_pose_start(task)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.distance_config = merge_nested(
            default_distance_config(task),
            load_json_arg(distance_config),
        )
        self.task_target_pos = target_pos_from_config(task, self.distance_config)
        self.distance_weights = self.distance_config["feature_weights"]
        self.pose_quat_format = str(
            self.distance_config.get("pose_quat_format", "xyzw")
        ).lower()
        self.relative_pose_mode = str(
            self.distance_config.get("relative_pose_mode", "eef_frame_object_delta")
        ).lower()
        self._load_dataset()
        self._build_tensors()
        self._build_pca_state_space()
        self.start_episode()
        print(
            f"Loaded abs-aligned KNN: task={task}, pairs={len(self.actions)}, "
            f"k={self.k}, action_horizon={self.action_horizon}, "
            f"future_progress={self.future_progress_mode}, "
            f"local_residual={self.local_residual}, "
            f"action_retarget={self.action_retarget}, flow_field={self.flow_field}, "
            f"knn_space={self.knn_space}, "
            f"device={self.device}"
        )

    def _load_dataset(self):
        states, actions, demo_indices, demo_positions = [], [], [], []
        demo_flat_indices = []
        demo_initial_object_xy = []
        with h5py.File(self.dataset_path, "r") as f:
            # Preserve HDF5 insertion order to keep KNN tie-breaking identical.
            demo_keys = [key for key in f["data"].keys() if key.startswith("demo_")]
            if (
                self.task == "can"
                and bool(self.distance_config.get("can_demo_target_from_demos", True))
                and demo_keys
            ):
                final_positions = []
                for demo_key in demo_keys:
                    if demo_key not in f["data"]:
                        continue
                    object_obs = np.asarray(f["data"][demo_key]["obs"]["object"])
                    object_pose = object_obs[-1, self.demo_object_pose_start:self.demo_object_pose_start + 3]
                    if object_pose.shape[0] == 3:
                        final_positions.append(object_pose.astype(np.float64))
                if final_positions:
                    final_positions = np.asarray(final_positions, dtype=np.float64)
                    percentile = float(
                        self.distance_config.get("can_demo_target_percentile", 50.0)
                    )
                    target_pos = np.percentile(final_positions, percentile, axis=0)
                    self.task_target_pos = target_pos.astype(np.float64)
                    self.distance_config["task_target_pos"] = (
                        self.task_target_pos.astype(float).tolist()
                    )
            for demo_idx, demo_key in enumerate(demo_keys):
                if demo_key not in f["data"]:
                    raise KeyError(f"{demo_key} is missing from abs dataset {self.dataset_path}")
                demo = f["data"][demo_key]
                obs_dict = {key: np.asarray(demo["obs"][key]) for key in RAW_OBS_KEYS}
                initial_object = obs_dict["object"][0, self.demo_object_pose_start:self.demo_object_pose_start + 2]
                demo_initial_object_xy.append(np.asarray(initial_object, dtype=np.float64))
                demo_actions = np.asarray(demo["actions"])
                current_demo_indices = []
                for t in range(len(demo_actions)):
                    flat_index = len(states)
                    state = state_from_demo_obs(
                        obs_dict,
                        t,
                        self.demo_object_pose_start,
                        target_pos=self.task_target_pos,
                        pose_quat_format=self.pose_quat_format,
                        relative_pose_mode=self.relative_pose_mode,
                    )
                    states.append(state)
                    actions.append(demo_actions[t])
                    demo_indices.append(demo_idx)
                    demo_positions.append(len(current_demo_indices))
                    current_demo_indices.append(flat_index)
                demo_flat_indices.append(current_demo_indices)
        self.state_components = states
        self.actions = np.asarray(actions, dtype=np.float32)
        self.demo_indices = np.asarray(demo_indices, dtype=np.int32)
        self.flat_demo_positions = np.asarray(demo_positions, dtype=np.int32)
        self.demo_flat_indices = demo_flat_indices
        self.demo_initial_object_xy = np.asarray(demo_initial_object_xy, dtype=np.float64)
        self._compute_norm_stats()
        self._calibrate_can_stage_thresholds_from_demos()
        self._calibrate_square_stage_thresholds_from_demos()
        self._compute_demo_phase_labels()
        self._compute_future_progress_values()

    def _compute_norm_stats(self):
        self.norm_stats = {}
        for key in OBS_KEYS:
            values = np.asarray([state[key] for state in self.state_components])
            self.norm_stats[key] = {
                "mean": values.mean(axis=0),
                "std": values.std(axis=0) + 1e-8,
            }





























    def start_episode(self):
        self._plan_indices = []
        self._plan_pointer = 0
        self._last_selection_info = {}
        self._last_action_info = {}
        self._last_object_abs_retarget_info = {}
        self._last_metric_info = {}
        self._last_future_progress_info = {}
        self._last_local_residual_info = {}
        self._last_metric_phase = None
        self._metric_phase_counts = {}
        self._vote_selected_count = 0
        self._future_progress_selected_count = 0
        self._local_residual_count = 0
        self._flow_counts = {"calls": 0}
        self._progress_demo = None
        self._progress_position = -1
        self._progress_guarded_count = 0
        self._episode_initial_object_z = None
        self._episode_initial_object_xy = None
        self._episode_initial_boundary_label = None
        self._episode_min_object_z = None
        self._episode_gripper_open_seen = False
        self._last_square_phase_id = None
        self._last_square_phase_info = {}
        self._last_can_phase_id = None
        self._last_can_phase_info = {}
        self._last_lift_phase_id = None
        self._last_lift_phase_info = {}
        self._can_lift_start_step = None
        self._phase_gate_counts = {}
        self._lift_start_preference_counts = {}
        self._initial_state_info = {}
        self._policy_step_count = 0

    def _task_key(self, suffix, task=None):
        task_name = self.task if task is None else task
        profile = TASK_GPI_PROFILES.get(task_name)
        if profile is None:
            return str(suffix)
        return f"{profile.name}_{suffix}"
