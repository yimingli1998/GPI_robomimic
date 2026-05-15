import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .config import *
from .state import normalize_quat, quat_xyzw_to_rotation, slerp_rotations, state_from_live_obs


class ActionMixin:
    def _compute_future_progress_values(self):
        n = len(self.state_components)
        self.future_xy_gain = np.zeros(n, dtype=np.float32)
        self.future_z_drop = np.zeros(n, dtype=np.float32)
        self.future_ori_gain = np.zeros(n, dtype=np.float32)
        self.future_success_score = np.zeros(n, dtype=np.float32)
        self.future_target_flat_index = np.arange(n, dtype=np.int64)
        self.future_terminal_xy = np.zeros(n, dtype=np.float32)
        self.future_terminal_z = np.zeros(n, dtype=np.float32)
        self.future_terminal_ori = np.zeros(n, dtype=np.float32)
        self.future_terminal_eef_obj = np.zeros(n, dtype=np.float32)
        if self.task not in {"square", "can"}:
            return

        horizon = max(1, int(self.future_progress_horizon))
        temp = max(1e-6, float(self.future_progress_success_temp))
        for demo_indices in self.demo_flat_indices:
            if not demo_indices:
                continue
            demo_indices_arr = np.asarray(demo_indices, dtype=np.int64)
            states = [self.state_components[int(idx)] for idx in demo_indices_arr]
            object_to_target = np.asarray([
                state["object_to_target_pos"] for state in states
            ], dtype=np.float64)
            object_z = np.asarray([
                state["object_pos"][2] for state in states
            ], dtype=np.float64)
            object_quat = np.asarray([
                normalize_quat(state["object_quat"]) for state in states
            ], dtype=np.float64)
            eef_obj = np.asarray([
                np.linalg.norm(state["relative_pos"]) for state in states
            ], dtype=np.float64)
            xy = np.linalg.norm(object_to_target[:, :2], axis=1)
            final_quat = normalize_quat(object_quat[-1])
            dots = np.abs(np.sum(object_quat * final_quat[None, :], axis=1))
            dots = np.clip(dots, -1.0, 1.0)
            ori = 2.0 * np.arccos(dots)
            length = len(demo_indices_arr)
            for pos, flat_index in enumerate(demo_indices_arr):
                end = min(length, pos + horizon + 1)
                future_pos = min(length - 1, pos + horizon)
                self.future_xy_gain[int(flat_index)] = float(xy[pos] - np.min(xy[pos:end]))
                self.future_z_drop[int(flat_index)] = float(object_z[pos] - np.min(object_z[pos:end]))
                self.future_ori_gain[int(flat_index)] = float(ori[pos] - np.min(ori[pos:end]))
                remaining = max(0, length - 1 - pos)
                self.future_success_score[int(flat_index)] = float(np.exp(-remaining / temp))
                self.future_target_flat_index[int(flat_index)] = int(demo_indices_arr[future_pos])
                self.future_terminal_xy[int(flat_index)] = float(xy[future_pos])
                self.future_terminal_z[int(flat_index)] = float(object_z[future_pos])
                self.future_terminal_ori[int(flat_index)] = float(ori[future_pos])
                self.future_terminal_eef_obj[int(flat_index)] = float(eef_obj[future_pos])

    def _plan_empty(self):
        return self._plan_pointer >= len(self._plan_indices)

    def _effective_action_horizon(self):
        return int(self.action_horizon)

    def _refresh_phase_during_plan(self, current_state):
        if self.task == "square" and not self._plan_empty():
            self._square_phase_gate_info(current_state)

    def _load_action_plan(self, flat_index, current_state=None):
        demo_idx = int(self.demo_indices[flat_index])
        demo_position = int(self.flat_demo_positions[flat_index])
        demo_indices = self.demo_flat_indices[demo_idx]
        action_horizon = self._effective_action_horizon()
        end = min(demo_position + action_horizon, len(demo_indices))
        self._plan_indices = [int(idx) for idx in demo_indices[demo_position:end]]
        self._plan_pointer = 0

    def _pop_plan_index(self):
        flat_index = self._plan_indices[self._plan_pointer]
        self._plan_pointer += 1
        return int(flat_index)

    def _resolved_retarget_mode(self, current_state):
        mode = self.action_retarget
        if mode != "phase_eefxyzw_object_abs":
            return mode
        if self._square_object_lifted_for_retarget(current_state):
            return "object_abs_delta_xyzw"
        return "object_frame_delta_xyzw"

    def _action_for_index(self, flat_index, current_state, env_state=None, override_mode=None):
        mode = override_mode or self._resolved_retarget_mode(current_state)
        self._last_resolved_mode = mode
        self._last_object_abs_retarget_info = {}
        action = self.actions[int(flat_index)].astype(np.float64).copy()
        if mode == "none":
            return action.astype(np.float32)

        demo_state = self.state_components[int(flat_index)]
        if mode == "object_abs_delta_xyzw":
            return self._retarget_object_abs_delta_xyzw(
                action,
                int(flat_index),
                demo_state,
                current_state,
            )

        if mode == "object_frame_delta_xyzw":
            return self._retarget_object_frame_delta_xyzw(
                action,
                demo_state,
                current_state,
            )

        if mode == "eef_pos_delta":
            return self._retarget_eef_pos_delta(action, demo_state, current_state)

        raise ValueError(f"Unknown action_retarget mode: {mode}")

    def _retarget_object_abs_delta_xyzw(self, action, flat_index, demo_state, current_state):
        demo_object_pos = np.asarray(demo_state["object_pos"], dtype=np.float64)
        demo_eef_pos = np.asarray(demo_state["robot0_eef_pos"], dtype=np.float64)
        current_object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
        current_eef_pos = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
        demo_object_rot = quat_xyzw_to_rotation(demo_state["object_quat"])
        current_object_rot = quat_xyzw_to_rotation(current_state["object_quat"])
        demo_eef_rot = quat_xyzw_to_rotation(demo_state["robot0_eef_quat"])
        current_eef_rot = quat_xyzw_to_rotation(current_state["robot0_eef_quat"])
        demo_target_eef_pos = np.asarray(action[:3], dtype=np.float64)
        demo_target_eef_rot = Rotation.from_rotvec(action[3:6])
        demo_object_to_eef_pos = demo_object_rot.inv().apply(demo_eef_pos - demo_object_pos)
        current_object_to_eef_pos = current_object_rot.inv().apply(
            current_eef_pos - current_object_pos
        )
        demo_object_to_eef_rot = demo_object_rot.inv() * demo_eef_rot
        current_object_to_eef_rot = current_object_rot.inv() * current_eef_rot
        source = str(
            self.distance_config.get(
                "square_object_abs_retarget_source",
                "action_implied",
            )
        )
        if source == "future_state":
            future_state = self._future_state_for_index(int(flat_index), 1)
            demo_object_target = np.asarray(future_state["object_pos"], dtype=np.float64)
            demo_object_target_rot = quat_xyzw_to_rotation(future_state["object_quat"])
        elif source == "action_implied":
            demo_object_target_rot = demo_target_eef_rot * demo_object_to_eef_rot.inv()
            demo_object_target = (
                demo_target_eef_pos
                - demo_object_target_rot.apply(demo_object_to_eef_pos)
            )
        else:
            raise ValueError(f"Unknown square_object_abs_retarget_source: {source}")

        eef_delta_target = current_eef_pos + (demo_target_eef_pos - demo_eef_pos)
        object_abs_target = (
            demo_object_target
            + demo_object_target_rot.apply(current_object_to_eef_pos)
        )
        blend = float(
            np.clip(
                float(self.distance_config.get("square_object_abs_retarget_blend", 1.0)),
                0.0,
                1.0,
            )
        )
        action[:3] = (1.0 - blend) * eef_delta_target + blend * object_abs_target

        orientation_blend = self._square_object_abs_orientation_blend(
            current_state,
            blend,
        )
        target_rot = slerp_rotations(
            demo_target_eef_rot * demo_eef_rot.inv() * current_eef_rot,
            demo_object_target_rot * current_object_to_eef_rot,
            orientation_blend,
        )
        action[3:6] = target_rot.as_rotvec()
        self._last_object_abs_retarget_info = {
            "mode": "object_abs_delta_xyzw",
            "source": source,
            "blend": float(blend),
            "orientation_blend": float(orientation_blend),
            "target_eef_pos": np.asarray(action[:3], dtype=np.float64).astype(float).tolist(),
            "target_eef_rotvec": np.asarray(action[3:6], dtype=np.float64).astype(float).tolist(),
        }
        return action.astype(np.float32)

    def _square_object_abs_orientation_blend(self, current_state, default_blend):
        orientation_blend = self.distance_config.get(
            "square_object_abs_orientation_blend",
            default_blend,
        )
        if orientation_blend is None:
            orientation_phase = self._square_after_lift_metric_phase(
                current_state,
                self.distance_config,
            )
            orientation_blend_key = {
                "square_pre_insert": "square_object_abs_orientation_pre_insert_blend",
                "square_insert": "square_object_abs_orientation_insert_blend",
            }.get(orientation_phase, "square_object_abs_orientation_transport_blend")
            orientation_blend = self.distance_config.get(
                orientation_blend_key,
                default_blend,
            )
        return float(np.clip(float(orientation_blend), 0.0, 1.0))

    def _retarget_object_frame_delta_xyzw(self, action, demo_state, current_state):
        demo_object_pos = np.asarray(demo_state["object_pos"], dtype=np.float64)
        current_object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
        demo_object_rot = quat_xyzw_to_rotation(demo_state["object_quat"])
        current_object_rot = quat_xyzw_to_rotation(current_state["object_quat"])
        object_delta = current_object_rot * demo_object_rot.inv()
        action[:3] = current_object_pos + object_delta.apply(action[:3] - demo_object_pos)
        demo_target_rot = Rotation.from_rotvec(action[3:6])
        action[3:6] = (object_delta * demo_target_rot).as_rotvec()
        return action.astype(np.float32)

    def _retarget_eef_pos_delta(self, action, demo_state, current_state):
        demo_eef_pos = np.asarray(demo_state["robot0_eef_pos"], dtype=np.float64)
        current_eef_pos = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
        action[:3] = current_eef_pos + (action[:3] - demo_eef_pos)
        return action.astype(np.float32)

    def _local_residual_active(self):
        if self.task != "square" or not self.local_residual:
            return False
        phase_info = getattr(self, "_last_square_phase_info", {}) or {}
        phase_id = phase_info.get("phase_id")
        if phase_id is None:
            return False
        return int(phase_id) >= int(self.local_residual_min_phase_id)

    def _apply_local_residual(self, action, flat_index, current_state):
        action = np.asarray(action, dtype=np.float64).copy()
        self._last_local_residual_info = {
            "active": False,
            "count": int(getattr(self, "_local_residual_count", 0)),
        }
        if not self._local_residual_active():
            return action.astype(np.float32)

        future_state = self._future_state_for_index(
            int(flat_index),
            int(self.local_residual_target_offset),
        )
        current_object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
        current_eef_pos = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
        current_object_rot = quat_xyzw_to_rotation(current_state["object_quat"])
        current_eef_rot = quat_xyzw_to_rotation(current_state["robot0_eef_quat"])
        current_object_to_eef_pos = current_object_rot.inv().apply(
            current_eef_pos - current_object_pos
        )
        current_object_to_eef_rot = current_object_rot.inv() * current_eef_rot

        target_object_pos = np.asarray(future_state["object_pos"], dtype=np.float64)
        target_object_rot = quat_xyzw_to_rotation(future_state["object_quat"])
        desired_eef_pos = (
            target_object_pos
            + target_object_rot.apply(current_object_to_eef_pos)
        )
        desired_eef_rot = target_object_rot * current_object_to_eef_rot

        raw_pos_correction = desired_eef_pos - action[:3]
        clipped_pos_correction = self._clip_vector_norm(
            raw_pos_correction,
            self.local_residual_max_pos,
        )
        action[:3] = (
            action[:3]
            + float(self.local_residual_pos_alpha) * clipped_pos_correction
        )

        action_rot = Rotation.from_rotvec(action[3:6])
        raw_rot_correction = (desired_eef_rot * action_rot.inv()).as_rotvec()
        clipped_rot_correction = self._clip_vector_norm(
            raw_rot_correction,
            self.local_residual_max_rot,
        )
        corrected_rot = (
            Rotation.from_rotvec(
                float(self.local_residual_ori_alpha) * clipped_rot_correction
            )
            * action_rot
        )
        action[3:6] = corrected_rot.as_rotvec()

        self._local_residual_count += 1
        self._last_local_residual_info = {
            "active": True,
            "count": int(self._local_residual_count),
            "phase": (getattr(self, "_last_square_phase_info", {}) or {}).get("phase"),
            "flat_index": int(flat_index),
            "target_offset": int(self.local_residual_target_offset),
            "target_flat_index": int(
                self.future_target_flat_index[int(flat_index)]
                if int(self.local_residual_target_offset) == int(self.future_progress_horizon)
                else self.demo_flat_indices[int(self.demo_indices[int(flat_index)])][
                    min(
                        int(self.flat_demo_positions[int(flat_index)])
                        + int(self.local_residual_target_offset),
                        len(self.demo_flat_indices[int(self.demo_indices[int(flat_index)])]) - 1,
                    )
                ]
            ),
            "pos_alpha": float(self.local_residual_pos_alpha),
            "ori_alpha": float(self.local_residual_ori_alpha),
            "max_pos": float(self.local_residual_max_pos),
            "max_rot": float(self.local_residual_max_rot),
            "raw_pos_correction_norm": float(np.linalg.norm(raw_pos_correction)),
            "applied_pos_correction": (
                float(self.local_residual_pos_alpha)
                * clipped_pos_correction
            ).astype(float).tolist(),
            "raw_rot_correction_norm": float(np.linalg.norm(raw_rot_correction)),
            "applied_rot_correction": (
                float(self.local_residual_ori_alpha)
                * clipped_rot_correction
            ).astype(float).tolist(),
        }
        return action.astype(np.float32)

    def _record_action_info(self, flat_index, action, selected_distance=None, from_plan=False):
        flat_index = int(flat_index)
        self._last_action_info = {
            "flat_index": flat_index,
            "demo": int(self.demo_indices[flat_index]),
            "demo_pos": int(self.flat_demo_positions[flat_index]),
            "selected_distance": None if selected_distance is None else float(selected_distance),
            "from_plan": bool(from_plan),
            "retarget_mode": getattr(self, "_last_resolved_mode", self.action_retarget),
            "plan_pointer": int(self._plan_pointer),
            "plan_length": int(len(self._plan_indices)),
            "progress_demo": None if self._progress_demo is None else int(self._progress_demo),
            "progress_position": int(self._progress_position),
            "progress_guarded_count": int(self._progress_guarded_count),
            "initial_state": self._initial_state_info,
            "action": np.asarray(action, dtype=np.float32).tolist(),
        }
        if self._last_metric_info:
            self._last_action_info["metric_info"] = self._last_metric_info
        if self._last_object_abs_retarget_info:
            self._last_action_info["object_abs_retarget"] = self._last_object_abs_retarget_info
        if self._last_local_residual_info:
            self._last_action_info["local_residual"] = self._last_local_residual_info

    def _blend_actions_for_indices(self, flat_indices, weights, current_state, env_state=None):
        actions = np.asarray([
            self._action_for_index(int(flat_index), current_state, env_state)
            for flat_index in flat_indices
        ])
        return np.sum(actions * weights[:, None], axis=0).astype(np.float32)

    def _topk_weights(self, distances, k):
        available = int(torch.isfinite(distances).sum().item())
        k = min(int(k), available)
        if k <= 0:
            raise RuntimeError("No finite KNN distances available")
        top_distances, top_indices = torch.topk(distances, k=k, largest=False)
        distances_np = top_distances.detach().cpu().numpy().astype(np.float64)
        indices_np = top_indices.detach().cpu().numpy().astype(np.int64)
        weights = 1.0 / (distances_np + 1e-6)
        weights = weights / np.sum(weights)
        return top_distances, top_indices, indices_np, weights.astype(np.float64)

    def _future_state_for_index(self, flat_index, offset):
        if offset <= 0:
            return self.state_components[int(flat_index)]
        demo_idx = int(self.demo_indices[int(flat_index)])
        demo_position = int(self.flat_demo_positions[int(flat_index)])
        demo_indices = self.demo_flat_indices[demo_idx]
        future_position = min(demo_position + int(offset), len(demo_indices) - 1)
        return self.state_components[int(demo_indices[future_position])]

    def _flow_target_position(self, current_state, neighbor_state):
        current_eef = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
        current_object = np.asarray(current_state["object_pos"], dtype=np.float64)
        neighbor_relative = np.asarray(neighbor_state["relative_pos"], dtype=np.float64)
        neighbor_eef = np.asarray(neighbor_state["robot0_eef_pos"], dtype=np.float64)

        if self.task == "can":
            gripper_open = bool(np.asarray(current_state["robot0_gripper_qpos"])[0] > 0.035)
            current_eef_rot = quat_xyzw_to_rotation(current_state["robot0_eef_quat"])
            return current_object - current_eef_rot.apply(neighbor_relative) if gripper_open else neighbor_eef

        if self._square_object_lifted_for_retarget(current_state):
            current_object_rot = quat_xyzw_to_rotation(current_state["object_quat"])
            neighbor_object_rot = quat_xyzw_to_rotation(neighbor_state["object_quat"])
            current_object_to_eef = current_object_rot.inv().apply(
                current_eef - current_object
            )
            object_abs_target = (
                np.asarray(neighbor_state["object_pos"], dtype=np.float64)
                + neighbor_object_rot.apply(current_object_to_eef)
            )
            blend = float(
                self.distance_config.get("square_object_abs_retarget_blend", 1.0)
            )
            blend = float(np.clip(blend, 0.0, 1.0))
            return (1.0 - blend) * neighbor_eef + blend * object_abs_target

        gripper_open = bool(np.asarray(current_state["robot0_gripper_qpos"])[0] > 0.035)
        current_eef_rot = quat_xyzw_to_rotation(current_state["robot0_eef_quat"])
        object_relative_target = current_object - current_eef_rot.apply(neighbor_relative)
        if gripper_open:
            return 0.7 * object_relative_target + 0.3 * neighbor_eef
        return neighbor_eef

    def _clip_vector_norm(self, vector, max_norm):
        vector = np.asarray(vector, dtype=np.float64)
        max_norm = float(max_norm)
        norm = float(np.linalg.norm(vector))
        if max_norm > 0.0 and norm > max_norm:
            vector = vector * (max_norm / (norm + 1e-8))
        return vector

    def _flow_action(
        self,
        distances,
        current_state,
        env_state=None,
        base_index=None,
        base_action=None,
        selected_index=None,
        selected_distance=None,
        from_plan=False,
    ):
        top_distances, _, indices_np, weights = self._topk_weights(distances, self.flow_neighbors)
        if selected_index is None:
            selected_index = int(indices_np[0])
        if base_index is None:
            base_index = int(selected_index)
        if base_action is None:
            base_action = self._action_for_index(int(base_index), current_state, env_state)

        action = np.asarray(base_action, dtype=np.float64).copy()
        current_eef = np.asarray(current_state["robot0_eef_pos"], dtype=np.float64)
        correction = np.zeros(3, dtype=np.float64)
        for weight, flat_index in zip(weights, indices_np):
            neighbor_state = self._future_state_for_index(int(flat_index), self.flow_target_offset)
            target_pos = self._flow_target_position(current_state, neighbor_state)
            correction += float(weight) * (target_pos - current_eef)
        correction = self._clip_vector_norm(correction, self.flow_attraction_clip)
        action[:3] = action[:3] + (self.flow_attraction - 1.0) * correction

        self._flow_counts["calls"] = self._flow_counts.get("calls", 0) + 1
        self._last_action_info = {
            "flat_index": int(base_index),
            "demo": int(self.demo_indices[int(base_index)]),
            "demo_pos": int(self.flat_demo_positions[int(base_index)]),
            "selected_distance": None if selected_distance is None else float(selected_distance),
            "from_plan": bool(from_plan),
            "current_nearest_index": int(selected_index),
            "current_nearest_demo": int(self.demo_indices[int(selected_index)]),
            "current_nearest_demo_pos": int(self.flat_demo_positions[int(selected_index)]),
            "retarget_mode": getattr(self, "_last_resolved_mode", self.action_retarget),
            "flow_field": True,
            "flow_active": True,
            "flow_neighbor_indices": indices_np.tolist(),
            "flow_neighbor_distances": top_distances.detach().cpu().numpy().astype(float).tolist(),
            "flow_neighbor_weights": weights.astype(float).tolist(),
            "flow_correction": correction.astype(float).tolist(),
            "flow_response_delta": (action[:3] - current_eef).astype(float).tolist(),
            "flow_attraction": float(self.flow_attraction),
            "flow_target_offset": int(self.flow_target_offset),
            "progress_demo": None if self._progress_demo is None else int(self._progress_demo),
            "progress_position": int(self._progress_position),
            "progress_guarded_count": int(self._progress_guarded_count),
            "initial_state": self._initial_state_info,
            "action": action.astype(np.float32).tolist(),
        }
        if self._last_metric_info:
            self._last_action_info["metric_info"] = self._last_metric_info
        if self._last_object_abs_retarget_info:
            self._last_action_info["object_abs_retarget"] = self._last_object_abs_retarget_info
        return action.astype(np.float32)

    def _flow_controller_enabled(self):
        return bool(self.flow_field)

    def _nearest_plan_start(self, distances):
        nearest_distance, nearest_index = torch.topk(distances, k=1, largest=False)
        return int(nearest_index.item()), float(nearest_distance.item())

    def _update_progress(self, flat_index):
        demo_idx = int(self.demo_indices[int(flat_index)])
        demo_position = int(self.flat_demo_positions[int(flat_index)])
        if self.progress_guard:
            if self._progress_demo == demo_idx:
                self._progress_position = max(self._progress_position, demo_position)
            else:
                self._progress_demo = demo_idx
                self._progress_position = demo_position

    def _progress_guard_distances(self, distances, current_state, record=True):
        if not self.progress_guard or self._progress_demo is None:
            return distances
        if self.progress_guard_closed_only:
            gripper_open = bool(np.asarray(current_state["robot0_gripper_qpos"])[0] > 0.035)
            if gripper_open:
                return distances
        if int(self._progress_position) < self.progress_guard_min_position:
            return distances
        min_position = max(0, int(self._progress_position) - self.progress_backtrack_window)
        stale = (
            (self.demo_index_tensor == int(self._progress_demo))
            & (self.demo_position_tensor < min_position)
        )
        if not bool(stale.any().item()):
            return distances
        guarded = distances.clone()
        guarded[stale] = float("inf")
        if record:
            self._progress_guarded_count += int(stale.sum().item())
        return guarded

    def _segment_stats(self, top_indices_np, top_distances_np, demo_ids, positions, center_index):
        center_demo = int(self.demo_indices[int(center_index)])
        center_position = int(self.flat_demo_positions[int(center_index)])
        same_segment = (
            (demo_ids == center_demo)
            & (np.abs(positions - center_position) <= self.vote_window)
        )
        group_indices = top_indices_np[same_segment]
        group_distances = top_distances_np[same_segment]
        if len(group_indices) == 0:
            return {
                "support": 1,
                "representative": int(center_index),
                "representative_distance": float("inf"),
                "score": 0.0,
            }
        representative = int(group_indices[int(np.argmin(group_distances))])
        representative_distance = float(np.min(group_distances))
        support = int(len(group_distances))
        return {
            "support": support,
            "representative": representative,
            "representative_distance": representative_distance,
            "score": float(support / (float(np.mean(group_distances)) + 1e-8)),
        }

    def _select_plan_start(self, distances, record_vote=True):
        selected_index, selected_distance = self._nearest_plan_start(distances)
        selected_score = 1.0 / (selected_distance + 1e-8)
        selected_support = 1

        available = int(torch.isfinite(distances).sum().item())
        top_n = min(self.vote_top_n, available)
        if top_n <= 1:
            return selected_index, selected_distance

        top_distances, top_indices = torch.topk(distances, k=top_n, largest=False)
        top_distances_np = top_distances.detach().cpu().numpy()
        top_indices_np = top_indices.detach().cpu().numpy().astype(np.int64)
        finite = np.isfinite(top_distances_np)
        if not np.any(finite):
            return selected_index, selected_distance
        top_distances_np = top_distances_np[finite]
        top_indices_np = top_indices_np[finite]
        demo_ids = self.demo_indices[top_indices_np]
        positions = self.flat_demo_positions[top_indices_np]

        selected_stats = self._segment_stats(
            top_indices_np, top_distances_np, demo_ids, positions, selected_index
        )
        selected_support = selected_stats["support"]
        selected_score = selected_stats["score"]

        best_score = -np.inf
        best_distance = selected_distance
        best_index = selected_index
        best_support = 0
        for center_index in top_indices_np:
            stats = self._segment_stats(
                top_indices_np, top_distances_np, demo_ids, positions, int(center_index)
            )
            if (
                stats["score"] > best_score
                or (np.isclose(stats["score"], best_score) and stats["representative_distance"] < best_distance)
            ):
                best_score = stats["score"]
                best_distance = stats["representative_distance"]
                best_index = stats["representative"]
                best_support = stats["support"]

        if best_index != selected_index:
            support_ok = best_support >= max(
                self.vote_min_support,
                selected_support + self.vote_support_margin,
            )
            score_ok = best_score >= selected_score * self.vote_score_ratio
            distance_ok = best_distance <= (
                selected_distance * self.vote_distance_ratio + self.vote_distance_margin
            )
            if support_ok and score_ok and distance_ok:
                if record_vote:
                    self._vote_selected_count += 1
                return best_index, best_distance
        return selected_index, selected_distance

    def _future_progress_active(self, phase_info):
        if self.task not in {"square", "can"} or self.future_progress_mode == "off":
            return False
        phase_id = (phase_info or {}).get("phase_id")
        if phase_id is None:
            return False
        return int(phase_id) >= int(self.future_progress_min_phase_id)

    def _future_progress_phase_params(self, phase_info):
        phase = None if phase_info is None else phase_info.get("phase")
        overrides = self.distance_config.get("future_progress_phase_overrides", {})
        phase_overrides = {}
        if isinstance(overrides, dict) and phase in overrides:
            raw = overrides.get(phase, {})
            if isinstance(raw, dict):
                phase_overrides = raw

        def value(name, default):
            if name in phase_overrides:
                return float(phase_overrides[name])
            prefixed = f"future_progress_{name}"
            if prefixed in phase_overrides:
                return float(phase_overrides[prefixed])
            return float(default)

        def optional_value(name, default):
            if name in phase_overrides:
                raw = phase_overrides[name]
            else:
                prefixed = f"future_progress_{name}"
                raw = phase_overrides.get(prefixed, default)
            if raw is None:
                return None
            return float(raw)

        return {
            "distance_weight": value(
                "distance_weight",
                self.future_progress_distance_weight,
            ),
            "xy_weight": value("xy_weight", self.future_progress_xy_weight),
            "z_weight": value("z_weight", self.future_progress_z_weight),
            "ori_weight": value("ori_weight", self.future_progress_ori_weight),
            "success_weight": value(
                "success_weight",
                self.future_progress_success_weight,
            ),
            "distance_ratio": value(
                "distance_ratio",
                self.future_progress_distance_ratio,
            ),
            "distance_margin": value(
                "distance_margin",
                self.future_progress_distance_margin,
            ),
            "terminal_xy_weight": value(
                "terminal_xy_weight",
                self.distance_config.get("future_progress_terminal_xy_weight", 0.0),
            ),
            "terminal_z_weight": value(
                "terminal_z_weight",
                self.distance_config.get("future_progress_terminal_z_weight", 0.0),
            ),
            "terminal_ori_weight": value(
                "terminal_ori_weight",
                self.distance_config.get("future_progress_terminal_ori_weight", 0.0),
            ),
            "terminal_eef_weight": value(
                "terminal_eef_weight",
                self.distance_config.get("future_progress_terminal_eef_weight", 0.0),
            ),
            "terminal_xy_scale": value(
                "terminal_xy_scale",
                self.distance_config.get("future_progress_terminal_xy_scale", 0.04),
            ),
            "terminal_z_scale": value(
                "terminal_z_scale",
                self.distance_config.get("future_progress_terminal_z_scale", 0.04),
            ),
            "terminal_ori_scale": value(
                "terminal_ori_scale",
                self.distance_config.get("future_progress_terminal_ori_scale", 0.8),
            ),
            "terminal_eef_scale": value(
                "terminal_eef_scale",
                self.distance_config.get("future_progress_terminal_eef_scale", 0.08),
            ),
            "terminal_z_target": optional_value(
                "terminal_z_target",
                self.distance_config.get("future_progress_terminal_z_target", None),
            ),
            "terminal_eef_target": optional_value(
                "terminal_eef_target",
                self.distance_config.get("future_progress_terminal_eef_target", None),
            ),
        }

    def _future_progress_scores(self, indices_np, distances_np, current_state=None, phase_info=None):
        distances_np = np.asarray(distances_np, dtype=np.float64)
        indices_np = np.asarray(indices_np, dtype=np.int64)
        phase_params = self._future_progress_phase_params(phase_info)
        best_distance = float(np.min(distances_np)) if len(distances_np) else 0.0
        distance_norm = np.log1p(
            np.maximum(0.0, distances_np - best_distance)
            / (abs(best_distance) + 1e-6)
        )
        xy_progress = np.clip(
            self.future_xy_gain[indices_np].astype(np.float64) / 0.04,
            -2.0,
            2.0,
        )
        z_progress = np.clip(
            self.future_z_drop[indices_np].astype(np.float64) / 0.04,
            -2.0,
            2.0,
        )
        ori_progress = np.clip(
            self.future_ori_gain[indices_np].astype(np.float64) / 0.35,
            -2.0,
            2.0,
        )
        success_score = self.future_success_score[indices_np].astype(np.float64)
        scores = (
            -float(phase_params["distance_weight"]) * distance_norm
            + float(phase_params["xy_weight"]) * xy_progress
            + float(phase_params["z_weight"]) * z_progress
            + float(phase_params["ori_weight"]) * ori_progress
            + float(phase_params["success_weight"]) * success_score
        )
        terminal_xy_weight = float(phase_params["terminal_xy_weight"])
        terminal_z_weight = float(phase_params["terminal_z_weight"])
        terminal_ori_weight = float(phase_params["terminal_ori_weight"])
        terminal_eef_weight = float(phase_params["terminal_eef_weight"])
        if (
            terminal_xy_weight != 0.0
            or terminal_z_weight != 0.0
            or terminal_ori_weight != 0.0
            or terminal_eef_weight != 0.0
        ):
            xy_scale = max(1e-6, float(phase_params["terminal_xy_scale"]))
            z_scale = max(1e-6, float(phase_params["terminal_z_scale"]))
            ori_scale = max(1e-6, float(phase_params["terminal_ori_scale"]))
            eef_scale = max(1e-6, float(phase_params["terminal_eef_scale"]))
            z_target = phase_params["terminal_z_target"]
            if z_target is None:
                z_target = float(
                    self.distance_config.get(
                        "square_phase_gate_insert_z_threshold",
                        self.distance_config.get("square_insert_z_threshold", 0.94),
                    )
                )
            eef_target = phase_params["terminal_eef_target"]
            if eef_target is None:
                eef_target = float(
                    self.distance_config.get("square_carry_eef_object_max", 0.11)
                )
            terminal_xy = np.clip(
                self.future_terminal_xy[indices_np].astype(np.float64) / xy_scale,
                0.0,
                5.0,
            )
            terminal_z = np.clip(
                np.maximum(
                    0.0,
                    self.future_terminal_z[indices_np].astype(np.float64) - float(z_target),
                )
                / z_scale,
                0.0,
                5.0,
            )
            terminal_ori = np.clip(
                self.future_terminal_ori[indices_np].astype(np.float64) / ori_scale,
                0.0,
                5.0,
            )
            terminal_eef = np.clip(
                np.maximum(
                    0.0,
                    self.future_terminal_eef_obj[indices_np].astype(np.float64)
                    - float(eef_target),
                )
                / eef_scale,
                0.0,
                5.0,
            )
            scores = (
                scores
                - terminal_xy_weight * terminal_xy
                - terminal_z_weight * terminal_z
                - terminal_ori_weight * terminal_ori
                - terminal_eef_weight * terminal_eef
            )
        return scores

    def _select_plan_start_with_future_progress(
        self,
        distances,
        current_state,
        phase_info,
        record_vote=True,
    ):
        base_index, base_distance = self._select_plan_start(
            distances,
            record_vote=record_vote,
        )
        self._last_future_progress_info = {
            "active": False,
            "mode": self.future_progress_mode,
            "count": int(getattr(self, "_future_progress_selected_count", 0)),
        }
        if not self._future_progress_active(phase_info):
            return base_index, base_distance

        available = int(torch.isfinite(distances).sum().item())
        top_n = min(int(self.future_progress_top_n), available)
        if top_n <= 1:
            return base_index, base_distance

        top_distances, top_indices = torch.topk(distances, k=top_n, largest=False)
        distances_np = top_distances.detach().cpu().numpy().astype(np.float64)
        indices_np = top_indices.detach().cpu().numpy().astype(np.int64)
        finite = np.isfinite(distances_np)
        distances_np = distances_np[finite]
        indices_np = indices_np[finite]
        if len(indices_np) <= 1:
            return base_index, base_distance

        scores = self._future_progress_scores(
            indices_np,
            distances_np,
            current_state=current_state,
            phase_info=phase_info,
        )
        score_by_index = {
            int(index): float(score)
            for index, score in zip(indices_np, scores)
        }
        if int(base_index) in score_by_index:
            base_score = float(score_by_index[int(base_index)])
        else:
            base_score = float(
                self._future_progress_scores(
                    np.asarray([int(base_index)], dtype=np.int64),
                    np.asarray([float(base_distance)], dtype=np.float64),
                    current_state=current_state,
                    phase_info=phase_info,
                )[0]
            )

        chosen_local = int(np.argmax(scores))

        candidate_index = int(indices_np[chosen_local])
        candidate_distance = float(distances_np[chosen_local])
        candidate_score = float(scores[chosen_local])
        phase_params = self._future_progress_phase_params(phase_info)
        distance_ok = candidate_distance <= (
            float(base_distance) * float(phase_params["distance_ratio"])
            + float(phase_params["distance_margin"])
        )
        score_ok = candidate_score > base_score
        accepted = bool(
            candidate_index != int(base_index)
            and score_ok
            and distance_ok
        )
        if accepted:
            self._future_progress_selected_count += 1
            selected_index = candidate_index
            selected_distance = candidate_distance
        else:
            selected_index = int(base_index)
            selected_distance = float(base_distance)

        top_items = []
        top_order = np.argsort(-scores)[: min(5, len(scores))]
        for local_i in top_order:
            flat_index = int(indices_np[int(local_i)])
            top_items.append({
                "flat_index": flat_index,
                "demo": int(self.demo_indices[flat_index]),
                "demo_pos": int(self.flat_demo_positions[flat_index]),
                "distance": float(distances_np[int(local_i)]),
                "score": float(scores[int(local_i)]),
                "xy_gain": float(self.future_xy_gain[flat_index]),
                "z_drop": float(self.future_z_drop[flat_index]),
                "ori_gain": float(self.future_ori_gain[flat_index]),
                "terminal_xy": float(self.future_terminal_xy[flat_index]),
                "terminal_z": float(self.future_terminal_z[flat_index]),
                "terminal_ori": float(self.future_terminal_ori[flat_index]),
                "terminal_eef_obj": float(self.future_terminal_eef_obj[flat_index]),
                "success_score": float(self.future_success_score[flat_index]),
            })
        self._last_future_progress_info = {
            "active": True,
            "mode": self.future_progress_mode,
            "accepted": bool(accepted),
            "count": int(self._future_progress_selected_count),
            "phase": None if phase_info is None else phase_info.get("phase"),
            "top_n": int(len(indices_np)),
            "base_index": int(base_index),
            "base_distance": float(base_distance),
            "base_score": float(base_score),
            "candidate_index": int(candidate_index),
            "candidate_distance": float(candidate_distance),
            "candidate_score": float(candidate_score),
            "selected_index": int(selected_index),
            "selected_distance": float(selected_distance),
            "distance_ok": bool(distance_ok),
            "score_ok": bool(score_ok),
            "phase_params": phase_params,
            "top_items": top_items,
        }
        return selected_index, selected_distance

    def _square_object_lifted_for_retarget(self, current_state, distance_config=None):
        if self.task != "square":
            return False
        distance_config = distance_config or self.distance_config
        object_z = float(np.asarray(current_state["object_pos"], dtype=np.float64)[2])
        if self._episode_min_object_z is None:
            self._episode_min_object_z = object_z
        else:
            self._episode_min_object_z = min(float(self._episode_min_object_z), object_z)
        table_z = float(self._episode_min_object_z)
        lift_delta = float(
            distance_config.get("square_phase_gate_pick_lift_delta", 0.03)
        )
        return bool(object_z >= table_z + lift_delta)

    def _square_after_lift_metric_phase(self, current_state, distance_config):
        if not self._square_object_lifted_for_retarget(current_state, distance_config):
            return None
        target_pos = target_pos_from_config(self.task, distance_config)
        if target_pos is None:
            return "square_transport"
        object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
        xy_dist = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        insert_xy = float(distance_config.get("square_insert_xy_threshold", 0.04))
        pre_insert_xy = float(distance_config.get("square_pre_insert_xy_threshold", 0.08))
        object_z = float(object_pos[2])
        insert_z = float(distance_config.get("square_insert_z_threshold", 0.94))
        if xy_dist <= insert_xy and object_z <= insert_z:
            return "square_insert"
        if xy_dist <= pre_insert_xy:
            return "square_pre_insert"
        return "square_transport"

    def _distances(self, current_state):
        return self._pca_state_distances(current_state)

    def _select_distances_and_plan_start(self, current_state):
        phase_info = self._gpi_phase_info(current_state)
        raw_distances = self._distances(current_state)
        if phase_info is not None:
            raw_distances = self._phase_gate_distances(
                raw_distances,
                phase_info,
                self.distance_config,
                record=True,
            )
            raw_distances = self._lift_start_preference_distances(
                raw_distances,
                phase_info,
                self.distance_config,
                record=True,
            )
        distances = self._progress_guard_distances(raw_distances, current_state)
        selected_index, selected_distance = self._select_plan_start_with_future_progress(
            distances,
            current_state,
            phase_info,
        )
        self._last_metric_info = {
            "metric_name": "base",
            "selected_index": int(selected_index),
            "selected_distance": float(selected_distance),
            "knn_space": self.knn_space,
            "task_profile": None if self.task_profile is None else self.task_profile.name,
            "phase_gate": self._current_task_phase_info(),
            "square_phase_gate": self._last_square_phase_info,
            "future_progress": self._last_future_progress_info,
        }
        return distances, selected_index, selected_distance

    def _current_state_from_obs(self, obs):
        return state_from_live_obs(
            obs,
            self.object_pose_start,
            target_pos=self.task_target_pos,
            pose_quat_format=self.pose_quat_format,
            relative_pose_mode=self.relative_pose_mode,
        )

    def _record_policy_step(self):
        self._policy_step_count += 1

    def _select_plan_index_for_horizon(self, selected_index, current_state):
        if self._effective_action_horizon() <= 1:
            return int(selected_index), False
        if self._plan_empty():
            self._load_action_plan(selected_index, current_state)
            from_plan = False
        else:
            from_plan = True
        return self._pop_plan_index(), from_plan

    def _flow_control_action(self, current_state, env_state=None):
        with torch.no_grad():
            distances, selected_index, selected_distance = self._select_distances_and_plan_start(
                current_state,
            )
            flat_index, from_plan = self._select_plan_index_for_horizon(
                selected_index,
                current_state,
            )
            self._update_progress(flat_index)
            action_base = self._action_for_index(flat_index, current_state, env_state)
            action_base = self._apply_local_residual(
                action_base,
                flat_index,
                current_state,
            )
            action = self._flow_action(
                distances,
                current_state,
                env_state=env_state,
                base_index=flat_index,
                base_action=action_base,
                selected_index=selected_index,
                selected_distance=selected_distance,
                from_plan=from_plan,
            )
            self._last_action_info["flow_reason"] = "global"
        self._record_policy_step()
        return action

    def _queued_plan_action(self, current_state, env_state=None):
        flat_index = self._pop_plan_index()
        action = self._action_for_index(flat_index, current_state, env_state)
        action = self._apply_local_residual(action, flat_index, current_state)
        self._update_progress(flat_index)
        self._record_action_info(
            flat_index,
            action,
            selected_distance=None,
            from_plan=True,
        )
        return action

    def _new_selection_action(self, current_state, env_state=None):
        with torch.no_grad():
            distances, selected_index, selected_distance = self._select_distances_and_plan_start(
                current_state,
            )
            if self._effective_action_horizon() > 1:
                self._load_action_plan(selected_index, current_state)
                flat_index = self._pop_plan_index()
                self._update_progress(flat_index)
                action = self._action_for_index(flat_index, current_state, env_state)
                action = self._apply_local_residual(action, flat_index, current_state)
                self._record_action_info(
                    flat_index,
                    action,
                    selected_distance=selected_distance,
                    from_plan=False,
                )
                return action, selected_distance

            k = min(self.k, int(torch.isfinite(distances).sum().item()))
            top_distances, top_indices = torch.topk(distances, k=k, largest=False)
            weights = 1.0 / (top_distances.detach().cpu().numpy() + 1e-8)
            weights = weights / weights.sum()
            action = self._blend_actions_for_indices(
                top_indices.detach().cpu().numpy(),
                weights,
                current_state,
                env_state,
            )
            flat_index = int(top_indices[0].item())
            action = self._apply_local_residual(action, flat_index, current_state)
            self._update_progress(flat_index)
            selected_distance = float(top_distances[0].item())
            self._record_action_info(
                flat_index,
                action,
                selected_distance=selected_distance,
                from_plan=False,
            )
            return action, selected_distance

    def __call__(self, obs, env_state=None):
        current_state = self._current_state_from_obs(obs)
        self._refresh_phase_during_plan(current_state)
        if self._flow_controller_enabled():
            return self._flow_control_action(current_state, env_state)
        if self._effective_action_horizon() > 1 and not self._plan_empty():
            return self._queued_plan_action(current_state, env_state)
        action, selected_distance = self._new_selection_action(current_state, env_state)
        self._record_policy_step()
        return action
