import numpy as np
import torch

from .config import *
from .state import first_sustained_true


class PhaseMixin:
    def _calibrate_can_stage_thresholds_from_demos(self):
        self.demo_can_stage_calibration = {}
        if (
            self.task != "can"
            or not bool(
                self.distance_config.get(
                    "can_demo_calibrate_stage_thresholds",
                    True,
                )
            )
        ):
            return

        final_xy = []
        for demo_indices in self.demo_flat_indices:
            if not demo_indices:
                continue
            final_state = self.state_components[int(demo_indices[-1])]
            object_to_target = np.asarray(
                final_state["object_to_target_pos"],
                dtype=np.float64,
            )
            final_xy.append(float(np.linalg.norm(object_to_target[:2])))
        if not final_xy:
            return

        final_xy = np.asarray(final_xy, dtype=np.float64)
        percentile = float(
            self.distance_config.get("can_demo_stage_transport_xy_percentile", 95.0)
        )
        margin = float(
            self.distance_config.get("can_demo_stage_transport_xy_margin", 0.010)
        )
        old_threshold = float(
            self.distance_config.get("can_transport_success_xy_threshold", 0.08)
        )
        calibrated = float(np.percentile(final_xy, percentile)) + margin
        self.distance_config["can_transport_success_xy_threshold"] = calibrated
        self.demo_can_stage_calibration = {
            "num_demos": int(len(final_xy)),
            "transport_xy_percentile": float(percentile),
            "transport_xy_margin": float(margin),
            "old_transport_success_xy_threshold": float(old_threshold),
            "transport_success_xy_threshold": float(calibrated),
            "demo_final_xy_min": float(np.min(final_xy)),
            "demo_final_xy_p50": float(np.percentile(final_xy, 50.0)),
            "demo_final_xy_p95": float(np.percentile(final_xy, 95.0)),
            "demo_final_xy_max": float(np.max(final_xy)),
        }

    def _square_demo_stage_events(self, distance_config=None):
        if self.task != "square":
            return []
        distance_config = distance_config or self.distance_config
        lift_delta = float(
            distance_config.get("square_phase_gate_pick_lift_delta", 0.03)
        )
        table_z_window = max(
            1,
            int(distance_config.get("square_phase_gate_table_z_window", 40)),
        )
        table_z_percentile = float(
            distance_config.get("square_phase_gate_table_z_percentile", 5.0)
        )
        lift_sustain_steps = max(
            1,
            int(distance_config.get("square_phase_gate_lift_sustain_steps", 3)),
        )
        transport_xy = float(
            distance_config.get("square_phase_gate_transport_xy_threshold", 0.16)
        )
        pre_insert_xy = float(
            distance_config.get("square_phase_gate_pre_insert_xy_threshold", 0.08)
        )
        insert_xy = float(
            distance_config.get("square_phase_gate_insert_xy_threshold", 0.04)
        )
        insert_z = float(
            distance_config.get(
                "square_phase_gate_insert_z_threshold",
                distance_config.get("square_insert_z_threshold", 0.94),
            )
        )

        events = []
        for demo_idx, demo_indices in enumerate(self.demo_flat_indices):
            if not demo_indices:
                continue
            arrays = self._demo_phase_arrays(demo_indices)
            object_z = arrays["object_z"]
            target_xy_dist = np.linalg.norm(arrays["object_to_target"][:, :2], axis=1)
            table_window = object_z[:min(len(object_z), table_z_window)]
            table_z = float(np.percentile(table_window, table_z_percentile))
            lifted = object_z >= table_z + lift_delta
            first_object_lift = first_sustained_true(
                lifted,
                sustain_steps=lift_sustain_steps,
            )
            final_idx = len(demo_indices) - 1
            event = {
                "demo_id": int(demo_idx),
                "length": int(len(demo_indices)),
                "table_z": float(table_z),
                "first_lift": None,
                "transport_pass": False,
                "pre_insert_pass": False,
                "insert_pass": False,
                "final_insert_pass": bool(
                    target_xy_dist[final_idx] <= insert_xy
                    and object_z[final_idx] <= insert_z
                ),
                "transport_xy_for_threshold": float(target_xy_dist[final_idx]),
                "pre_insert_xy_for_threshold": float(target_xy_dist[final_idx]),
                "insert_xy_for_threshold": float(target_xy_dist[final_idx]),
                "insert_z_for_threshold": float(object_z[final_idx]),
                "final_xy": float(target_xy_dist[final_idx]),
                "final_z": float(object_z[final_idx]),
            }
            if first_object_lift is None:
                events.append(event)
                continue

            first_object_lift = int(first_object_lift)
            event["first_lift"] = first_object_lift
            positions = np.arange(len(demo_indices))
            after_lift = positions >= first_object_lift
            min_after_lift_idx = first_object_lift + int(
                np.argmin(target_xy_dist[first_object_lift:])
            )

            transport_candidates = np.flatnonzero(
                after_lift & (target_xy_dist <= transport_xy)
            )
            if len(transport_candidates):
                first_transport = int(transport_candidates[0])
                event["transport_pass"] = True
            else:
                first_transport = int(min_after_lift_idx)
            event["first_transport"] = int(first_transport)
            event["transport_xy_for_threshold"] = float(
                target_xy_dist[first_transport]
            )

            pre_insert_candidates = np.flatnonzero(
                (positions >= first_transport)
                & (target_xy_dist <= pre_insert_xy)
            )
            if len(pre_insert_candidates):
                first_pre_insert = int(pre_insert_candidates[0])
                event["pre_insert_pass"] = True
            else:
                first_pre_insert = first_transport + int(
                    np.argmin(target_xy_dist[first_transport:])
                )
            event["first_pre_insert"] = int(first_pre_insert)
            event["pre_insert_xy_for_threshold"] = float(
                target_xy_dist[first_pre_insert]
            )

            insert_candidates = np.flatnonzero(
                (positions >= first_pre_insert)
                & (target_xy_dist <= insert_xy)
                & (object_z <= insert_z)
            )
            if len(insert_candidates):
                first_insert = int(insert_candidates[0])
                event["insert_pass"] = True
            else:
                first_insert = int(final_idx)
            event["first_insert"] = int(first_insert)
            event["insert_xy_for_threshold"] = float(
                max(target_xy_dist[first_insert], target_xy_dist[final_idx])
            )
            event["insert_z_for_threshold"] = float(
                max(object_z[first_insert], object_z[final_idx])
            )
            events.append(event)
        return events

    def _calibrate_square_stage_thresholds_from_demos(self):
        self.demo_stage_calibration = {}
        if (
            self.task != "square"
            or not bool(
                self.distance_config.get(
                    "square_demo_calibrate_stage_thresholds",
                    True,
                )
            )
        ):
            return

        events = self._square_demo_stage_events(self.distance_config)
        if not events:
            return

        old_thresholds = {
            "transport_xy": float(
                self.distance_config.get(
                    "square_phase_gate_transport_xy_threshold",
                    0.16,
                )
            ),
            "pre_insert_xy": float(
                self.distance_config.get(
                    "square_phase_gate_pre_insert_xy_threshold",
                    0.08,
                )
            ),
            "insert_xy": float(
                self.distance_config.get("square_phase_gate_insert_xy_threshold", 0.04)
            ),
            "insert_z": float(
                self.distance_config.get(
                    "square_phase_gate_insert_z_threshold",
                    self.distance_config.get("square_insert_z_threshold", 0.94),
                )
            ),
        }
        margins = {
            "transport_xy": float(
                self.distance_config.get("square_demo_stage_transport_xy_margin", 0.005)
            ),
            "pre_insert_xy": float(
                self.distance_config.get("square_demo_stage_pre_insert_xy_margin", 0.003)
            ),
            "insert_xy": float(
                self.distance_config.get("square_demo_stage_insert_xy_margin", 0.002)
            ),
            "insert_z": float(
                self.distance_config.get("square_demo_stage_insert_z_margin", 0.002)
            ),
        }

        corner_values = {
            "transport_xy": float(max(
                event["transport_xy_for_threshold"] for event in events
            )),
            "pre_insert_xy": float(max(
                event["pre_insert_xy_for_threshold"] for event in events
            )),
            "insert_xy": float(max(
                event["insert_xy_for_threshold"] for event in events
            )),
            "insert_z": float(max(
                event["insert_z_for_threshold"] for event in events
            )),
        }
        calibrated = {
            "transport_xy": corner_values["transport_xy"] + margins["transport_xy"],
            "pre_insert_xy": corner_values["pre_insert_xy"] + margins["pre_insert_xy"],
            "insert_xy": corner_values["insert_xy"] + margins["insert_xy"],
            "insert_z": corner_values["insert_z"] + margins["insert_z"],
        }
        calibrated["pre_insert_xy"] = max(
            calibrated["pre_insert_xy"],
            calibrated["insert_xy"],
        )
        calibrated["transport_xy"] = max(
            calibrated["transport_xy"],
            calibrated["pre_insert_xy"],
        )

        self.distance_config["square_phase_gate_transport_xy_threshold"] = float(
            calibrated["transport_xy"]
        )
        self.distance_config["square_phase_gate_pre_insert_xy_threshold"] = float(
            calibrated["pre_insert_xy"]
        )
        self.distance_config["square_phase_gate_insert_xy_threshold"] = float(
            calibrated["insert_xy"]
        )
        self.distance_config["square_phase_gate_insert_z_threshold"] = float(
            calibrated["insert_z"]
        )
        self.distance_config["square_pre_insert_xy_threshold"] = float(
            calibrated["pre_insert_xy"]
        )
        self.distance_config["square_insert_xy_threshold"] = float(
            calibrated["insert_xy"]
        )
        self.distance_config["square_insert_z_threshold"] = float(
            calibrated["insert_z"]
        )

        validation_events = self._square_demo_stage_events(self.distance_config)
        pass_counts = {
            "lift": int(sum(event["first_lift"] is not None for event in validation_events)),
            "transport": int(sum(event["transport_pass"] for event in validation_events)),
            "pre_insert": int(sum(event["pre_insert_pass"] for event in validation_events)),
            "insert": int(sum(event["insert_pass"] for event in validation_events)),
            "final_insert": int(sum(event["final_insert_pass"] for event in validation_events)),
        }
        failed_demo_ids = {
            stage: [
                int(event["demo_id"])
                for event in validation_events
                if not bool(event[flag])
            ]
            for stage, flag in {
                "transport": "transport_pass",
                "pre_insert": "pre_insert_pass",
                "insert": "insert_pass",
                "final_insert": "final_insert_pass",
            }.items()
        }
        failed_demo_ids["lift"] = [
            int(event["demo_id"])
            for event in validation_events
            if event["first_lift"] is None
        ]
        self.demo_stage_calibration = {
            "num_demos": int(len(validation_events)),
            "old_thresholds": old_thresholds,
            "demo_corner_values": corner_values,
            "margins": margins,
            "calibrated_thresholds": {
                key: float(value) for key, value in calibrated.items()
            },
            "pass_counts": pass_counts,
            "failed_demo_ids": {
                stage: ids for stage, ids in failed_demo_ids.items() if ids
            },
        }

    def _demo_phase_arrays(self, demo_indices):
        states = [self.state_components[int(idx)] for idx in demo_indices]
        object_pos = np.asarray([state["object_pos"] for state in states], dtype=np.float64)
        eef_pos = np.asarray([state["robot0_eef_pos"] for state in states], dtype=np.float64)
        gripper_qpos = np.asarray([
            state["robot0_gripper_qpos"][0] for state in states
        ], dtype=np.float64)
        return {
            "states": states,
            "object_pos": object_pos,
            "object_to_target": np.asarray([
                state["object_to_target_pos"] for state in states
            ], dtype=np.float64),
            "eef_pos": eef_pos,
            "gripper_qpos": gripper_qpos,
            "object_z": object_pos[:, 2],
            "eef_object_dist": np.linalg.norm(eef_pos - object_pos, axis=1),
        }

    def _record_lift_samples(
        self,
        lift_success_samples,
        lift_start_samples,
        demo_indices,
        first_object_lift,
        lift_start_backtrack,
        object_z,
        table_z,
        eef_object_dist,
        gripper_qpos,
    ):
        if first_object_lift >= len(demo_indices):
            return
        success_pos = int(first_object_lift)
        lift_success_samples.append(
            self._lift_sample(
                demo_indices,
                success_pos,
                object_z,
                table_z,
                eef_object_dist,
                gripper_qpos,
            )
        )
        lift_start_begin = max(0, success_pos - lift_start_backtrack)
        for sample_pos in range(lift_start_begin, success_pos + 1):
            lift_start_samples.append(
                self._lift_sample(
                    demo_indices,
                    sample_pos,
                    object_z,
                    table_z,
                    eef_object_dist,
                    gripper_qpos,
                )
            )

    def _lift_sample(
        self,
        demo_indices,
        sample_pos,
        object_z,
        table_z,
        eef_object_dist,
        gripper_qpos,
    ):
        state = self.state_components[int(demo_indices[sample_pos])]
        return {
            "object_lift": float(object_z[sample_pos] - table_z),
            "eef_object_dist": float(eef_object_dist[sample_pos]),
            "gripper_qpos": float(gripper_qpos[sample_pos]),
            "object_frame_relative_pos": np.asarray(
                state["object_frame_relative_pos"],
                dtype=np.float64,
            ),
            "robot0_gripper_qpos": np.asarray(
                state["robot0_gripper_qpos"],
                dtype=np.float64,
            ),
        }

    def _finish_demo_phase_labels(
        self,
        phase_ids,
        phase_step_ids,
        lift_start_window,
        phase_name_to_id,
        min_std_key,
    ):
        self.demo_phase_ids = phase_ids
        self.demo_phase_step_ids = phase_step_ids
        self.demo_lift_start_window = lift_start_window
        self.demo_lift_start_window_tensor = torch.as_tensor(
            self.demo_lift_start_window,
            dtype=torch.bool,
            device=self.device,
        )
        self.demo_phase_tensor = torch.as_tensor(
            self.demo_phase_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.demo_phase_step_tensor = torch.as_tensor(
            self.demo_phase_step_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.demo_phase_counts = {
            phase_name: int(np.sum(phase_ids == phase_id))
            for phase_name, phase_id in phase_name_to_id.items()
        }
        self.demo_lift_start_window_count = int(np.sum(lift_start_window))
        self.demo_lift_start_profile = self._lift_start_profile(
            lift_start_window,
            min_std_key,
        )

    def _lift_start_profile(self, lift_start_window, min_std_key):
        if int(np.sum(lift_start_window)) <= 0:
            return {}
        lift_start_states = [
            self.state_components[idx]
            for idx in np.flatnonzero(lift_start_window)
        ]
        relative_values = np.asarray([
            state["object_frame_relative_pos"] for state in lift_start_states
        ], dtype=np.float64)
        gripper_values = np.asarray([
            state["robot0_gripper_qpos"] for state in lift_start_states
        ], dtype=np.float64)
        min_std = float(self.distance_config.get(min_std_key, 0.01))
        return {
            "object_frame_relative_pos_mean": relative_values.mean(axis=0),
            "object_frame_relative_pos_std": np.maximum(
                relative_values.std(axis=0),
                min_std,
            ),
            "robot0_gripper_qpos_mean": gripper_values.mean(axis=0),
            "robot0_gripper_qpos_std": np.maximum(
                gripper_values.std(axis=0),
                min_std,
            ),
        }

    def _compute_demo_phase_labels(self):
        phase_ids = np.zeros(len(self.state_components), dtype=np.int64)
        phase_step_ids = np.zeros(len(self.state_components), dtype=np.int64)
        lift_start_window = np.zeros(len(self.state_components), dtype=bool)
        if self.task in {"can", "lift"}:
            profile = self.task_profile
            prefix = profile.name
            phase_name_to_id = profile.phase_ids
            lift_success_samples = []
            lift_start_samples = []

            lift_delta = float(
                self.distance_config.get(f"{prefix}_phase_gate_pick_lift_delta", 0.03)
            )
            table_z_window = max(
                1,
                int(self.distance_config.get(f"{prefix}_phase_gate_table_z_window", 40)),
            )
            table_z_percentile = float(
                self.distance_config.get(f"{prefix}_phase_gate_table_z_percentile", 5.0)
            )
            lift_sustain_steps = max(
                1,
                int(self.distance_config.get(f"{prefix}_phase_gate_lift_sustain_steps", 3)),
            )
            lift_start_backtrack = max(
                0,
                int(self.distance_config.get(f"{prefix}_phase_gate_lift_start_backtrack", 4)),
            )
            lift_start_contact_max = float(
                self.distance_config.get(
                    f"{prefix}_phase_gate_lift_start_contact_eef_object_max",
                    0.085,
                )
            )
            lift_start_open_threshold = float(
                self.distance_config.get(
                    f"{prefix}_phase_gate_lift_start_gripper_open_threshold",
                    0.035,
                )
            )
            lift_start_close_threshold = float(
                self.distance_config.get(
                    f"{prefix}_phase_gate_lift_start_gripper_close_threshold",
                    0.03,
                )
            )
            lift_start_window_after = max(
                0,
                int(self.distance_config.get(f"{prefix}_lift_start_preference_window_after", 12)),
            )
            lift_window = max(
                1,
                int(self.distance_config.get(f"{prefix}_phase_gate_lift_window", 24)),
            )

            for demo_indices in self.demo_flat_indices:
                if not demo_indices:
                    continue
                arrays = self._demo_phase_arrays(demo_indices)
                object_z = arrays["object_z"]
                gripper_qpos = arrays["gripper_qpos"]
                eef_object_dist = arrays["eef_object_dist"]
                table_window = object_z[:min(len(object_z), table_z_window)]
                table_z = float(np.percentile(table_window, table_z_percentile))
                open_seen = np.maximum.accumulate(
                    gripper_qpos > lift_start_open_threshold
                )
                close_contact = (
                    open_seen
                    & (gripper_qpos <= lift_start_close_threshold)
                    & (eef_object_dist <= lift_start_contact_max)
                )
                first_close = first_sustained_true(close_contact, sustain_steps=1)
                lifted = object_z >= table_z + lift_delta
                first_object_lift = first_sustained_true(
                    lifted,
                    sustain_steps=lift_sustain_steps,
                )
                if first_object_lift is None:
                    first_object_lift = len(demo_indices)
                object_lift_backtrack = max(
                    0,
                    int(first_object_lift) - lift_start_backtrack,
                )
                first_lift = (
                    min(
                        max(0, int(first_close) - lift_start_backtrack),
                        object_lift_backtrack,
                    )
                    if first_close is not None
                    else object_lift_backtrack
                )

                self._record_lift_samples(
                    lift_success_samples,
                    lift_start_samples,
                    demo_indices,
                    int(first_object_lift),
                    lift_start_backtrack,
                    object_z,
                    table_z,
                    eef_object_dist,
                    gripper_qpos,
                )

                if self.task == "can":
                    first_transport = (
                        min(len(demo_indices), int(first_object_lift) + lift_window)
                        if first_object_lift < len(demo_indices)
                        else len(demo_indices)
                    )
                else:
                    first_transport = len(demo_indices)
                lift_start_end = min(
                    len(demo_indices),
                    max(first_lift + 1, int(first_object_lift) + lift_start_window_after),
                )
                for flat_index in demo_indices[first_lift:lift_start_end]:
                    lift_start_window[int(flat_index)] = True

                for local_pos, flat_index in enumerate(demo_indices):
                    if local_pos < first_lift:
                        phase_id = phase_name_to_id["pick"]
                        phase_start = 0
                    elif self.task != "can" or local_pos < first_transport:
                        phase_id = phase_name_to_id["lift"]
                        phase_start = first_lift
                    else:
                        phase_id = phase_name_to_id["transport"]
                        phase_start = first_transport
                    phase_ids[int(flat_index)] = phase_id
                    phase_step_ids[int(flat_index)] = max(0, int(local_pos - phase_start))

            self._finish_demo_phase_labels(
                phase_ids,
                phase_step_ids,
                lift_start_window,
                phase_name_to_id,
                f"{prefix}_lift_start_preference_min_std",
            )
            self._calibrate_pick_lift_thresholds(
                prefix,
                lift_success_samples,
                lift_start_samples,
            )
            return

        if self.task != "square":
            self.demo_phase_ids = phase_ids
            self.demo_phase_step_ids = phase_step_ids
            self.demo_lift_start_window = lift_start_window
            self.demo_lift_calibration = {}
            self.demo_lift_success_thresholds = {}
            self.demo_lift_start_window_tensor = torch.as_tensor(
                self.demo_lift_start_window,
                dtype=torch.bool,
                device=self.device,
            )
            self.demo_phase_tensor = torch.as_tensor(
                self.demo_phase_ids,
                dtype=torch.long,
                device=self.device,
            )
            self.demo_phase_step_tensor = torch.as_tensor(
                self.demo_phase_step_ids,
                dtype=torch.long,
                device=self.device,
            )
            self.demo_phase_counts = {"none": int(len(phase_ids))}
            return

        lift_success_samples = []
        lift_start_samples = []

        lift_delta = float(
            self.distance_config.get("square_phase_gate_pick_lift_delta", 0.03)
        )
        table_z_window = max(
            1,
            int(self.distance_config.get("square_phase_gate_table_z_window", 40)),
        )
        table_z_percentile = float(
            self.distance_config.get("square_phase_gate_table_z_percentile", 5.0)
        )
        lift_sustain_steps = max(
            1,
            int(self.distance_config.get("square_phase_gate_lift_sustain_steps", 3)),
        )
        lift_start_backtrack = max(
            0,
            int(self.distance_config.get("square_phase_gate_lift_start_backtrack", 4)),
        )
        lift_start_contact_max = float(
            self.distance_config.get(
                "square_phase_gate_lift_start_contact_eef_object_max",
                0.085,
            )
        )
        lift_start_open_threshold = float(
            self.distance_config.get(
                "square_phase_gate_lift_start_gripper_open_threshold",
                0.035,
            )
        )
        lift_start_close_threshold = float(
            self.distance_config.get(
                "square_phase_gate_lift_start_gripper_close_threshold",
                0.03,
            )
        )
        lift_start_window_after = max(
            0,
            int(self.distance_config.get("square_lift_start_preference_window_after", 12)),
        )
        lift_window = max(
            1,
            int(self.distance_config.get("square_phase_gate_lift_window", 24)),
        )
        transport_xy = float(
            self.distance_config.get("square_phase_gate_transport_xy_threshold", 0.16)
        )
        pre_insert_xy = float(
            self.distance_config.get("square_phase_gate_pre_insert_xy_threshold", 0.08)
        )
        insert_xy = float(
            self.distance_config.get("square_phase_gate_insert_xy_threshold", 0.04)
        )
        insert_z = float(
            self.distance_config.get(
                "square_phase_gate_insert_z_threshold",
                self.distance_config.get("square_insert_z_threshold", 0.94),
            )
        )

        for demo_indices in self.demo_flat_indices:
            if not demo_indices:
                continue
            arrays = self._demo_phase_arrays(demo_indices)
            object_z = arrays["object_z"]
            gripper_qpos = arrays["gripper_qpos"]
            eef_object_dist = arrays["eef_object_dist"]
            target_xy_dist = np.linalg.norm(arrays["object_to_target"][:, :2], axis=1)
            table_window = object_z[:min(len(object_z), table_z_window)]
            table_z = float(np.percentile(table_window, table_z_percentile))
            open_seen = np.maximum.accumulate(gripper_qpos > lift_start_open_threshold)
            close_contact = (
                open_seen
                & (gripper_qpos <= lift_start_close_threshold)
                & (eef_object_dist <= lift_start_contact_max)
            )
            first_close = first_sustained_true(close_contact, sustain_steps=1)
            lifted = object_z >= table_z + lift_delta
            first_object_lift = first_sustained_true(
                lifted,
                sustain_steps=lift_sustain_steps,
            )
            if first_object_lift is None:
                first_object_lift = len(demo_indices)
            object_lift_backtrack = max(
                0,
                int(first_object_lift) - lift_start_backtrack,
            )
            first_lift = (
                min(
                    max(0, int(first_close) - lift_start_backtrack),
                    object_lift_backtrack,
                )
                if first_close is not None
                else object_lift_backtrack
            )

            self._record_lift_samples(
                lift_success_samples,
                lift_start_samples,
                demo_indices,
                int(first_object_lift),
                lift_start_backtrack,
                object_z,
                table_z,
                eef_object_dist,
                gripper_qpos,
            )

            if first_object_lift < len(demo_indices):
                transport_candidates = np.flatnonzero(
                    (np.arange(len(demo_indices)) >= first_object_lift)
                    & (target_xy_dist <= transport_xy)
                )
                first_transport_event = (
                    int(transport_candidates[0])
                    if len(transport_candidates)
                    else len(demo_indices)
                )
                first_transport = min(
                    len(demo_indices),
                    max(first_object_lift + lift_window, first_transport_event),
                )
            else:
                first_transport = len(demo_indices)

            lift_start_end = min(
                len(demo_indices),
                max(first_lift + 1, int(first_object_lift) + lift_start_window_after),
            )
            for flat_index in demo_indices[first_lift:lift_start_end]:
                lift_start_window[int(flat_index)] = True

            pre_insert_candidates = np.flatnonzero(
                (np.arange(len(demo_indices)) >= first_transport)
                & (target_xy_dist <= pre_insert_xy)
            )
            first_pre_insert = (
                int(pre_insert_candidates[0])
                if len(pre_insert_candidates)
                else len(demo_indices)
            )

            seated_insert_candidates = np.flatnonzero(
                (np.arange(len(demo_indices)) >= first_pre_insert)
                & (target_xy_dist <= insert_xy)
                & (object_z <= insert_z)
            )
            first_insert = (
                int(seated_insert_candidates[0])
                if len(seated_insert_candidates)
                else len(demo_indices)
            )

            for local_pos, flat_index in enumerate(demo_indices):
                if local_pos < first_lift:
                    phase_id = SQUARE_PHASE_IDS["pick"]
                    phase_start = 0
                elif local_pos < first_transport:
                    phase_id = SQUARE_PHASE_IDS["lift"]
                    phase_start = first_lift
                elif local_pos < first_pre_insert:
                    phase_id = SQUARE_PHASE_IDS["transport"]
                    phase_start = first_transport
                elif local_pos < first_insert:
                    phase_id = SQUARE_PHASE_IDS["pre_insert"]
                    phase_start = first_pre_insert
                else:
                    phase_id = SQUARE_PHASE_IDS["insert"]
                    phase_start = first_insert
                phase_ids[int(flat_index)] = phase_id
                phase_step_ids[int(flat_index)] = max(0, int(local_pos - phase_start))

        self._finish_demo_phase_labels(
            phase_ids,
            phase_step_ids,
            lift_start_window,
            SQUARE_PHASE_IDS,
            "square_lift_start_preference_min_std",
        )
        self._calibrate_square_lift_thresholds(
            lift_success_samples,
            lift_start_samples,
        )

    def _calibrate_square_lift_thresholds(self, lift_success_samples, lift_start_samples):
        self.demo_lift_calibration = {}
        self.demo_lift_success_thresholds = {}
        if (
            self.task != "square"
            or not bool(self.distance_config.get("square_demo_calibrate_lift_thresholds", True))
            or not lift_success_samples
        ):
            return

        success_lifts = np.asarray([
            sample["object_lift"] for sample in lift_success_samples
        ], dtype=np.float64)
        success_dists = np.asarray([
            sample["eef_object_dist"] for sample in lift_success_samples
        ], dtype=np.float64)
        success_grippers = np.asarray([
            sample["gripper_qpos"] for sample in lift_success_samples
        ], dtype=np.float64)

        start_samples = lift_start_samples or lift_success_samples
        start_dists = np.asarray([
            sample["eef_object_dist"] for sample in start_samples
        ], dtype=np.float64)
        start_grippers = np.asarray([
            sample["gripper_qpos"] for sample in start_samples
        ], dtype=np.float64)

        lift_margin = float(
            self.distance_config.get("square_demo_lift_success_lift_margin", 0.002)
        )
        success_eef_margin = float(
            self.distance_config.get("square_demo_lift_success_eef_margin", 0.005)
        )
        contact_margin = float(
            self.distance_config.get("square_demo_lift_start_contact_margin", 0.005)
        )
        gripper_margin = float(
            self.distance_config.get("square_demo_lift_start_gripper_margin", 0.003)
        )
        sigma_margin = float(
            self.distance_config.get("square_demo_lift_start_sigma_margin", 0.25)
        )

        calibrated_lift_delta = max(0.0, float(np.min(success_lifts)) - lift_margin)
        calibrated_success_eef = float(np.max(success_dists)) + success_eef_margin
        calibrated_contact = float(np.max(start_dists)) + contact_margin
        calibrated_close = float(np.max(start_grippers)) + gripper_margin

        calibration = {
            "num_lift_success_samples": int(len(lift_success_samples)),
            "num_lift_start_samples": int(len(start_samples)),
            "success_object_lift_min": float(np.min(success_lifts)),
            "success_object_lift_max": float(np.max(success_lifts)),
            "success_eef_object_dist_max": float(np.max(success_dists)),
            "success_gripper_qpos_max": float(np.max(success_grippers)),
            "lift_start_eef_object_dist_max": float(np.max(start_dists)),
            "lift_start_gripper_qpos_max": float(np.max(start_grippers)),
            "pick_success_lift_delta": float(calibrated_lift_delta),
            "pick_success_eef_object_max": float(calibrated_success_eef),
            "lift_start_contact_eef_object_max": float(calibrated_contact),
            "lift_start_gripper_close_threshold": float(calibrated_close),
            "margins": {
                "lift": float(lift_margin),
                "success_eef": float(success_eef_margin),
                "contact": float(contact_margin),
                "gripper": float(gripper_margin),
                "sigma": float(sigma_margin),
            },
        }

        profile = getattr(self, "demo_lift_start_profile", {})
        if profile and start_samples:
            relative_scores = []
            gripper_scores = []
            for sample in start_samples:
                relative = np.asarray(
                    sample["object_frame_relative_pos"],
                    dtype=np.float64,
                )
                gripper = np.asarray(
                    sample["robot0_gripper_qpos"],
                    dtype=np.float64,
                )
                relative_scores.append(float(np.linalg.norm(
                    (
                        relative
                        - profile["object_frame_relative_pos_mean"]
                    )
                    / profile["object_frame_relative_pos_std"]
                )))
                gripper_scores.append(float(np.linalg.norm(
                    (
                        gripper
                        - profile["robot0_gripper_qpos_mean"]
                    )
                    / profile["robot0_gripper_qpos_std"]
                )))
            calibration["lift_start_relative_sigma_max"] = (
                float(np.max(relative_scores)) + sigma_margin
            )
            calibration["lift_start_gripper_sigma_max"] = (
                float(np.max(gripper_scores)) + sigma_margin
            )
            calibration["lift_start_relative_sigma_demo_max"] = float(np.max(relative_scores))
            calibration["lift_start_gripper_sigma_demo_max"] = float(np.max(gripper_scores))
            self.distance_config["square_lift_start_preference_relative_sigma_max"] = (
                calibration["lift_start_relative_sigma_max"]
            )
            self.distance_config["square_lift_start_preference_gripper_sigma_max"] = (
                calibration["lift_start_gripper_sigma_max"]
            )

        self.distance_config["square_phase_gate_lift_start_contact_eef_object_max"] = (
            calibration["lift_start_contact_eef_object_max"]
        )
        self.distance_config["square_phase_gate_lift_start_gripper_close_threshold"] = (
            calibration["lift_start_gripper_close_threshold"]
        )
        self.distance_config["square_phase_gate_pick_lift_delta"] = (
            calibration["pick_success_lift_delta"]
        )
        self.demo_lift_success_thresholds = {
            "lift_delta": calibration["pick_success_lift_delta"],
            "eef_object_max": calibration["pick_success_eef_object_max"],
        }
        self.demo_lift_calibration = calibration

    def _calibrate_pick_lift_thresholds(
        self,
        prefix,
        lift_success_samples,
        lift_start_samples,
    ):
        self.demo_lift_calibration = {}
        self.demo_lift_success_thresholds = {}
        if (
            self.task != prefix
            or not bool(
                self.distance_config.get(f"{prefix}_demo_calibrate_lift_thresholds", True)
            )
            or not lift_success_samples
        ):
            return

        success_lifts = np.asarray([
            sample["object_lift"] for sample in lift_success_samples
        ], dtype=np.float64)
        success_dists = np.asarray([
            sample["eef_object_dist"] for sample in lift_success_samples
        ], dtype=np.float64)
        success_grippers = np.asarray([
            sample["gripper_qpos"] for sample in lift_success_samples
        ], dtype=np.float64)

        start_samples = lift_start_samples or lift_success_samples
        start_dists = np.asarray([
            sample["eef_object_dist"] for sample in start_samples
        ], dtype=np.float64)
        start_grippers = np.asarray([
            sample["gripper_qpos"] for sample in start_samples
        ], dtype=np.float64)

        lift_margin = float(
            self.distance_config.get(f"{prefix}_demo_lift_success_lift_margin", 0.002)
        )
        success_eef_margin = float(
            self.distance_config.get(f"{prefix}_demo_lift_success_eef_margin", 0.005)
        )
        contact_margin = float(
            self.distance_config.get(f"{prefix}_demo_lift_start_contact_margin", 0.005)
        )
        gripper_margin = float(
            self.distance_config.get(f"{prefix}_demo_lift_start_gripper_margin", 0.003)
        )
        sigma_margin = float(
            self.distance_config.get(f"{prefix}_demo_lift_start_sigma_margin", 0.25)
        )

        calibrated_lift_delta = max(0.0, float(np.min(success_lifts)) - lift_margin)
        calibrated_success_eef = float(np.max(success_dists)) + success_eef_margin
        calibrated_contact = float(np.max(start_dists)) + contact_margin
        calibrated_close = float(np.max(start_grippers)) + gripper_margin

        calibration = {
            "num_lift_success_samples": int(len(lift_success_samples)),
            "num_lift_start_samples": int(len(start_samples)),
            "success_object_lift_min": float(np.min(success_lifts)),
            "success_object_lift_max": float(np.max(success_lifts)),
            "success_eef_object_dist_max": float(np.max(success_dists)),
            "success_gripper_qpos_max": float(np.max(success_grippers)),
            "lift_start_eef_object_dist_max": float(np.max(start_dists)),
            "lift_start_gripper_qpos_max": float(np.max(start_grippers)),
            "pick_success_lift_delta": float(calibrated_lift_delta),
            "pick_success_eef_object_max": float(calibrated_success_eef),
            "lift_start_contact_eef_object_max": float(calibrated_contact),
            "lift_start_gripper_close_threshold": float(calibrated_close),
            "margins": {
                "lift": float(lift_margin),
                "success_eef": float(success_eef_margin),
                "contact": float(contact_margin),
                "gripper": float(gripper_margin),
                "sigma": float(sigma_margin),
            },
        }

        profile = getattr(self, "demo_lift_start_profile", {})
        if profile and start_samples:
            relative_scores = []
            gripper_scores = []
            for sample in start_samples:
                relative = np.asarray(
                    sample["object_frame_relative_pos"],
                    dtype=np.float64,
                )
                gripper = np.asarray(
                    sample["robot0_gripper_qpos"],
                    dtype=np.float64,
                )
                relative_scores.append(float(np.linalg.norm(
                    (
                        relative
                        - profile["object_frame_relative_pos_mean"]
                    )
                    / profile["object_frame_relative_pos_std"]
                )))
                gripper_scores.append(float(np.linalg.norm(
                    (
                        gripper
                        - profile["robot0_gripper_qpos_mean"]
                    )
                    / profile["robot0_gripper_qpos_std"]
                )))
            calibration["lift_start_relative_sigma_max"] = (
                float(np.max(relative_scores)) + sigma_margin
            )
            calibration["lift_start_gripper_sigma_max"] = (
                float(np.max(gripper_scores)) + sigma_margin
            )
            calibration["lift_start_relative_sigma_demo_max"] = float(np.max(relative_scores))
            calibration["lift_start_gripper_sigma_demo_max"] = float(np.max(gripper_scores))
            self.distance_config[f"{prefix}_lift_start_preference_relative_sigma_max"] = (
                calibration["lift_start_relative_sigma_max"]
            )
            self.distance_config[f"{prefix}_lift_start_preference_gripper_sigma_max"] = (
                calibration["lift_start_gripper_sigma_max"]
            )

        self.distance_config[f"{prefix}_phase_gate_lift_start_contact_eef_object_max"] = (
            calibration["lift_start_contact_eef_object_max"]
        )
        self.distance_config[f"{prefix}_phase_gate_lift_start_gripper_close_threshold"] = (
            calibration["lift_start_gripper_close_threshold"]
        )
        self.distance_config[f"{prefix}_phase_gate_pick_lift_delta"] = (
            calibration["pick_success_lift_delta"]
        )
        self.demo_lift_success_thresholds = {
            "lift_delta": calibration["pick_success_lift_delta"],
            "eef_object_max": calibration["pick_success_eef_object_max"],
        }
        self.demo_lift_calibration = calibration

    def _current_task_phase_info(self):
        if self.task == "square":
            return getattr(self, "_last_square_phase_info", {}) or {}
        if self.task == "can":
            return getattr(self, "_last_can_phase_info", {}) or {}
        if self.task == "lift":
            return getattr(self, "_last_lift_phase_info", {}) or {}
        return {}

    def _record_task_phase_info(self, phase_payload):
        if self.task == "can":
            self._last_can_phase_info = phase_payload
        elif self.task == "lift":
            self._last_lift_phase_info = phase_payload
        if self.task in {"can", "square"}:
            self._last_square_phase_info = phase_payload

    def _gpi_phase_info(self, current_state):
        if self.task == "square":
            return self._square_phase_gate_info(current_state)
        if self.task in {"can", "lift"}:
            return self._can_phase_gate_info(current_state)
        return None

    def _gpi_phase_gate_context(self, phase_info, distance_config):
        if self.task == "square":
            return self._square_phase_gate_context(phase_info, distance_config)
        if self.task in {"can", "lift"}:
            return self._can_phase_gate_context(phase_info, distance_config)
        return None

    def _square_raw_phase_id(self, current_state, distance_config):
        if self.task != "square":
            return 0
        object_pos = np.asarray(current_state["object_pos"], dtype=np.float64)
        object_z = float(object_pos[2])
        if self._episode_initial_object_z is None:
            self._episode_initial_object_z = object_z
        if self._episode_min_object_z is None:
            self._episode_min_object_z = object_z
        else:
            self._episode_min_object_z = min(float(self._episode_min_object_z), object_z)
        table_z = float(self._episode_min_object_z)
        lift_delta = float(
            distance_config.get("square_phase_gate_pick_lift_delta", 0.03)
        )
        if object_z < table_z + lift_delta:
            return SQUARE_PHASE_IDS["pick"]

        object_to_target = np.asarray(
            current_state["object_to_target_pos"],
            dtype=np.float64,
        )
        xy_dist = float(np.linalg.norm(object_to_target[:2]))
        pre_insert_xy = float(
            distance_config.get("square_phase_gate_pre_insert_xy_threshold", 0.08)
        )
        insert_xy = float(
            distance_config.get("square_phase_gate_insert_xy_threshold", 0.04)
        )
        insert_z = float(
            distance_config.get(
                "square_phase_gate_insert_z_threshold",
                distance_config.get("square_insert_z_threshold", 0.94),
            )
        )
        transport_xy = float(
            distance_config.get("square_phase_gate_transport_xy_threshold", 0.16)
        )
        if xy_dist <= insert_xy and object_z <= insert_z:
            return SQUARE_PHASE_IDS["insert"]
        if xy_dist <= pre_insert_xy:
            return SQUARE_PHASE_IDS["pre_insert"]
        if xy_dist <= transport_xy:
            return SQUARE_PHASE_IDS["transport"]
        return SQUARE_PHASE_IDS["lift"]

    def _square_lift_start_preference_ready(self, current_state, distance_config):
        if self.task != "square":
            return False, {}
        object_z = float(np.asarray(current_state["object_pos"], dtype=np.float64)[2])
        table_z = object_z if self._episode_min_object_z is None else float(self._episode_min_object_z)
        lift_delta = float(
            distance_config.get("square_phase_gate_pick_lift_delta", 0.03)
        )
        if object_z >= table_z + lift_delta:
            return False, {"reason": "already_lifted"}

        gripper_qpos = float(
            np.asarray(current_state["robot0_gripper_qpos"], dtype=np.float64)[0]
        )
        open_threshold = float(
            distance_config.get(
                "square_phase_gate_lift_start_gripper_open_threshold",
                0.035,
            )
        )
        close_threshold = float(
            distance_config.get(
                "square_phase_gate_lift_start_gripper_close_threshold",
                0.03,
            )
        )
        contact_max = float(
            distance_config.get(
                "square_phase_gate_lift_start_contact_eef_object_max",
                0.085,
            )
        )
        if gripper_qpos > open_threshold:
            self._episode_gripper_open_seen = True
        eef_object_dist = float(
            np.linalg.norm(np.asarray(current_state["relative_pos"], dtype=np.float64))
        )
        threshold_ready = bool(
            self._episode_gripper_open_seen
            and gripper_qpos <= close_threshold
            and eef_object_dist <= contact_max
        )
        info = {
            "threshold_ready": bool(threshold_ready),
            "gripper_open_seen": bool(self._episode_gripper_open_seen),
            "gripper_qpos": float(gripper_qpos),
            "eef_object_dist": float(eef_object_dist),
        }
        profile = getattr(self, "demo_lift_start_profile", {})
        if not profile:
            return bool(threshold_ready), info

        relative = np.asarray(
            current_state["object_frame_relative_pos"],
            dtype=np.float64,
        )
        gripper = np.asarray(
            current_state["robot0_gripper_qpos"],
            dtype=np.float64,
        )
        relative_score = float(np.linalg.norm(
            (relative - profile["object_frame_relative_pos_mean"])
            / profile["object_frame_relative_pos_std"]
        ))
        gripper_score = float(np.linalg.norm(
            (gripper - profile["robot0_gripper_qpos_mean"])
            / profile["robot0_gripper_qpos_std"]
        ))
        relative_limit = float(
            distance_config.get(
                "square_lift_start_preference_relative_sigma_max",
                4.0,
            )
        )
        gripper_limit = float(
            distance_config.get(
                "square_lift_start_preference_gripper_sigma_max",
                3.0,
            )
        )
        gripper_profile_ready = bool(gripper_score <= gripper_limit)
        profile_ready = bool(
            relative_score <= relative_limit
            and gripper_profile_ready
        )
        info.update({
            "profile_ready": bool(profile_ready),
            "gripper_profile_ready": bool(gripper_profile_ready),
            "relative_profile_score": float(relative_score),
            "relative_profile_limit": float(relative_limit),
            "gripper_profile_score": float(gripper_score),
            "gripper_profile_limit": float(gripper_limit),
        })
        return bool(threshold_ready and profile_ready), info

    def _can_lift_start_preference_ready(self, current_state, distance_config):
        if self.task not in {"can", "lift"}:
            return False, {}
        prefix = self.task
        object_z = float(np.asarray(current_state["object_pos"], dtype=np.float64)[2])
        table_z = object_z if self._episode_min_object_z is None else float(self._episode_min_object_z)
        lift_delta = float(
            distance_config.get(f"{prefix}_phase_gate_pick_lift_delta", 0.03)
        )
        if object_z >= table_z + lift_delta:
            return False, {"reason": "already_lifted"}

        gripper_qpos = float(
            np.asarray(current_state["robot0_gripper_qpos"], dtype=np.float64)[0]
        )
        open_threshold = float(
            distance_config.get(
                f"{prefix}_phase_gate_lift_start_gripper_open_threshold",
                0.035,
            )
        )
        close_threshold = float(
            distance_config.get(
                f"{prefix}_phase_gate_lift_start_gripper_close_threshold",
                0.03,
            )
        )
        contact_max = float(
            distance_config.get(
                f"{prefix}_phase_gate_lift_start_contact_eef_object_max",
                0.085,
            )
        )
        if gripper_qpos > open_threshold:
            self._episode_gripper_open_seen = True
        eef_object_dist = float(
            np.linalg.norm(np.asarray(current_state["relative_pos"], dtype=np.float64))
        )
        threshold_ready = bool(
            self._episode_gripper_open_seen
            and gripper_qpos <= close_threshold
            and eef_object_dist <= contact_max
        )
        relaxed_contact_max_raw = distance_config.get(
            f"{prefix}_lift_start_relaxed_contact_max",
            None,
        )
        relaxed_contact_max = (
            contact_max
            if relaxed_contact_max_raw is None
            else float(relaxed_contact_max_raw)
        )
        relaxed_gripper_max = float(
            distance_config.get(
                f"{prefix}_lift_start_relaxed_gripper_max",
                close_threshold,
            )
        )
        relaxed_ready = bool(
            distance_config.get(f"{prefix}_lift_start_relaxed_ready", False)
            and self._episode_gripper_open_seen
            and eef_object_dist <= relaxed_contact_max
            and gripper_qpos <= relaxed_gripper_max
        )
        info = {
            "threshold_ready": bool(threshold_ready),
            "relaxed_ready": bool(relaxed_ready),
            "gripper_open_seen": bool(self._episode_gripper_open_seen),
            "gripper_qpos": float(gripper_qpos),
            "eef_object_dist": float(eef_object_dist),
            "relaxed_contact_max": float(relaxed_contact_max),
            "relaxed_gripper_max": float(relaxed_gripper_max),
        }
        profile = getattr(self, "demo_lift_start_profile", {})
        if not profile:
            return bool(threshold_ready or relaxed_ready), info

        relative = np.asarray(
            current_state["object_frame_relative_pos"],
            dtype=np.float64,
        )
        gripper = np.asarray(
            current_state["robot0_gripper_qpos"],
            dtype=np.float64,
        )
        relative_score = float(np.linalg.norm(
            (relative - profile["object_frame_relative_pos_mean"])
            / profile["object_frame_relative_pos_std"]
        ))
        gripper_score = float(np.linalg.norm(
            (gripper - profile["robot0_gripper_qpos_mean"])
            / profile["robot0_gripper_qpos_std"]
        ))
        relative_limit = float(
            distance_config.get(
                f"{prefix}_lift_start_preference_relative_sigma_max",
                4.0,
            )
        )
        gripper_limit = float(
            distance_config.get(
                f"{prefix}_lift_start_preference_gripper_sigma_max",
                3.0,
            )
        )
        ignore_gripper_profile = bool(
            distance_config.get(
                f"{prefix}_lift_start_relaxed_ignore_gripper_profile",
                True,
            )
        )
        gripper_profile_ready = bool(
            gripper_score <= gripper_limit
            or (relaxed_ready and ignore_gripper_profile)
        )
        profile_ready = bool(
            relative_score <= relative_limit
            and gripper_profile_ready
        )
        info.update({
            "profile_ready": bool(profile_ready),
            "gripper_profile_ready": bool(gripper_profile_ready),
            "relative_profile_score": float(relative_score),
            "relative_profile_limit": float(relative_limit),
            "gripper_profile_score": float(gripper_score),
            "gripper_profile_limit": float(gripper_limit),
        })
        return bool((threshold_ready or relaxed_ready) and profile_ready), info

    def _can_raw_phase_id(self, current_state, distance_config):
        if self.task not in {"can", "lift"}:
            return 0
        prefix = self.task
        phase_ids = self.task_profile.phase_ids
        object_z = float(np.asarray(current_state["object_pos"], dtype=np.float64)[2])
        if self._episode_initial_object_z is None:
            self._episode_initial_object_z = object_z
        if self._episode_min_object_z is None:
            self._episode_min_object_z = object_z
        else:
            self._episode_min_object_z = min(float(self._episode_min_object_z), object_z)
        table_z = float(self._episode_min_object_z)
        lift_delta = float(
            distance_config.get(f"{prefix}_phase_gate_pick_lift_delta", 0.03)
        )
        if object_z < table_z + lift_delta:
            self._can_lift_start_step = None
            return phase_ids["pick"]

        if self._can_lift_start_step is None:
            self._can_lift_start_step = int(self._policy_step_count)
        if self.task == "lift":
            return phase_ids["lift"]
        lift_window = max(
            1,
            int(distance_config.get(f"{prefix}_phase_gate_lift_window", 24)),
        )
        lift_steps = int(self._policy_step_count) - int(self._can_lift_start_step)
        if lift_steps < lift_window:
            return phase_ids["lift"]
        return phase_ids["transport"]

    def _can_phase_gate_info(self, current_state):
        profile = self.task_profile
        phase_names = list(profile.phase_names)
        phase_ids = profile.phase_ids
        raw_phase_id = self._can_raw_phase_id(current_state, self.distance_config)
        lift_start_ready, lift_start_ready_info = self._can_lift_start_preference_ready(
            current_state,
            self.distance_config,
        )
        last_phase_attr = f"_last_{self.task}_phase_id"
        previous_phase_id = getattr(self, last_phase_attr, None)
        if raw_phase_id == phase_ids["pick"]:
            phase_id = raw_phase_id
        elif previous_phase_id is None:
            phase_id = raw_phase_id
        else:
            phase_id = max(
                previous_phase_id,
                min(raw_phase_id, previous_phase_id + 1),
            )
        setattr(self, last_phase_attr, phase_id)
        object_to_target = np.asarray(
            current_state["object_to_target_pos"],
            dtype=np.float64,
        )
        info = {
            "phase_id": int(phase_id),
            "phase": phase_names[int(phase_id)],
            "raw_phase_id": int(raw_phase_id),
            "raw_phase": phase_names[int(raw_phase_id)],
            "previous_phase_id": (
                None if previous_phase_id is None else int(previous_phase_id)
            ),
            "previous_phase": (
                None
                if previous_phase_id is None
                else phase_names[int(previous_phase_id)]
            ),
            "object_z": float(np.asarray(current_state["object_pos"])[2]),
            "initial_object_z": None
            if self._episode_initial_object_z is None
            else float(self._episode_initial_object_z),
            "table_object_z": None
            if self._episode_min_object_z is None
            else float(self._episode_min_object_z),
            "object_to_target_xy": float(np.linalg.norm(object_to_target[:2])),
            "lift_start_step": (
                None if self._can_lift_start_step is None else int(self._can_lift_start_step)
            ),
            "lift_start_ready": bool(lift_start_ready),
            "lift_start_ready_info": lift_start_ready_info,
        }
        self._record_task_phase_info(info)
        return info

    def _square_phase_gate_info(self, current_state):
        raw_phase_id = self._square_raw_phase_id(current_state, self.distance_config)
        lift_start_ready, lift_start_ready_info = self._square_lift_start_preference_ready(
            current_state,
            self.distance_config,
        )
        previous_phase_id = self._last_square_phase_id
        if raw_phase_id == SQUARE_PHASE_IDS["pick"]:
            phase_id = raw_phase_id
        elif previous_phase_id is None:
            phase_id = raw_phase_id
        else:
            phase_id = max(
                previous_phase_id,
                min(raw_phase_id, previous_phase_id + 1),
            )
        self._last_square_phase_id = phase_id
        object_to_target = np.asarray(
            current_state["object_to_target_pos"],
            dtype=np.float64,
        )
        info = {
            "phase_id": int(phase_id),
            "phase": SQUARE_PHASE_NAMES[int(phase_id)],
            "raw_phase_id": int(raw_phase_id),
            "raw_phase": SQUARE_PHASE_NAMES[int(raw_phase_id)],
            "previous_phase_id": (
                None if previous_phase_id is None else int(previous_phase_id)
            ),
            "previous_phase": (
                None
                if previous_phase_id is None
                else SQUARE_PHASE_NAMES[int(previous_phase_id)]
            ),
            "object_z": float(np.asarray(current_state["object_pos"])[2]),
            "initial_object_z": None
            if self._episode_initial_object_z is None
            else float(self._episode_initial_object_z),
            "table_object_z": None
            if self._episode_min_object_z is None
            else float(self._episode_min_object_z),
            "object_to_target_xy": float(np.linalg.norm(object_to_target[:2])),
            "lift_start_ready": bool(lift_start_ready),
            "lift_start_ready_info": lift_start_ready_info,
        }
        self._record_task_phase_info(info)
        return info

    def _square_phase_gate_context(self, phase_info, distance_config):
        fine_phase_id = int(phase_info["phase_id"])
        return {
            "granularity": "fine",
            "prefix": "square",
            "phase_id": fine_phase_id,
            "phase": SQUARE_PHASE_NAMES[fine_phase_id],
            "phase_names": SQUARE_PHASE_NAMES,
            "phase_name_to_id": SQUARE_PHASE_IDS,
            "phase_tensor": self.demo_phase_tensor,
            "phase_step_tensor": self.demo_phase_step_tensor,
            "insert_id": SQUARE_PHASE_IDS["insert"],
            "pre_insert_id": SQUARE_PHASE_IDS["pre_insert"],
        }

    def _can_phase_gate_context(self, phase_info, distance_config):
        phase_id = int(phase_info["phase_id"])
        profile = self.task_profile
        prefix = profile.name
        phase_names = list(profile.phase_names)
        phase_name_to_id = profile.phase_ids
        terminal_id = phase_name_to_id.get("transport", phase_name_to_id["lift"])
        return {
            "granularity": "fine",
            "prefix": prefix,
            "phase_id": phase_id,
            "phase": phase_names[phase_id],
            "phase_names": phase_names,
            "phase_name_to_id": phase_name_to_id,
            "phase_tensor": self.demo_phase_tensor,
            "phase_step_tensor": self.demo_phase_step_tensor,
            "insert_id": terminal_id,
            "pre_insert_id": phase_name_to_id["lift"],
        }

    def _phase_name_to_context_id(self, phase_name, context, default_id):
        phase_name = str(phase_name)
        if phase_name in context["phase_name_to_id"]:
            return int(context["phase_name_to_id"][phase_name])
        return int(default_id)

    def _allowed_phase_ids_for_context(self, context, distance_config, phase_info=None):
        phase_id = int(context["phase_id"])
        prefix = str(context.get("prefix", "square"))
        allowed = [phase_id]
        if bool(distance_config.get(f"{prefix}_phase_gate_allow_next", True)):
            allowed.append(min(phase_id + 1, len(context["phase_names"]) - 1))
        return sorted(set(int(item) for item in allowed))

    def _phase_gate_distances(self, distances, phase_info, distance_config, record=True):
        profile = self.task_profile
        if profile is None:
            return distances
        prefix = profile.name
        if not bool(distance_config.get(self._task_key("phase_gate_metric"), False)):
            return distances
        context = self._gpi_phase_gate_context(phase_info, distance_config)
        if context is None:
            return distances
        mode = str(distance_config.get(f"{prefix}_phase_gate_mode", "hard")).lower()
        min_phase = distance_config.get(f"{prefix}_phase_gate_min_phase", None)
        if min_phase is not None:
            min_phase_id = self._phase_name_to_context_id(
                str(min_phase),
                context,
                0,
            )
            if int(context["phase_id"]) < int(min_phase_id):
                return distances
        if mode in {"soft", "penalty"}:
            min_phase = str(
                distance_config.get(
                    f"{prefix}_phase_gate_soft_min_phase",
                    context["phase_names"][0],
                )
            )
            max_phase = str(
                distance_config.get(
                    f"{prefix}_phase_gate_soft_max_phase",
                    context["phase_names"][-1],
                )
            )
            min_phase_id = self._phase_name_to_context_id(
                min_phase,
                context,
                0,
            )
            max_phase_id = self._phase_name_to_context_id(
                max_phase,
                context,
                len(context["phase_names"]) - 1,
            )
            current_phase_id = int(context["phase_id"])
            if int(min_phase_id) <= current_phase_id <= int(max_phase_id):
                return self._soft_phase_gate_distances(
                    distances,
                    phase_info,
                    distance_config,
                    context,
                    record=record,
                )
        allowed_ids = self._allowed_phase_ids_for_context(
            context,
            distance_config,
            phase_info=phase_info,
        )
        phase_tensor = context["phase_tensor"]
        phase_step_tensor = context["phase_step_tensor"]
        allowed_mask = torch.zeros_like(phase_tensor, dtype=torch.bool)
        current_phase_id = int(context["phase_id"])
        next_window = distance_config.get(f"{prefix}_phase_gate_next_window", None)
        next_window_value = None
        if next_window is not None:
            next_window_value = max(0, int(next_window))
        for phase_id in allowed_ids:
            phase_id = int(phase_id)
            phase_mask = phase_tensor == phase_id
            if (
                next_window_value is not None
                and phase_id != current_phase_id
            ):
                if next_window_value <= 0:
                    phase_mask = torch.zeros_like(phase_mask, dtype=torch.bool)
                else:
                    phase_mask = phase_mask & (
                        phase_step_tensor < next_window_value
                    )
            allowed_mask = allowed_mask | phase_mask
        if not bool(allowed_mask.any().item()):
            return distances

        gated = distances.clone()
        gated[~allowed_mask] = float("inf")
        finite_after = int(torch.isfinite(gated).sum().item())
        if finite_after <= 0 and bool(
            distance_config.get(f"{prefix}_phase_gate_fallback_to_all", True)
        ):
            return distances

        if record:
            key = f"{self.task}:{context['granularity']}:{context['phase']}"
            self._phase_gate_counts[key] = self._phase_gate_counts.get(key, 0) + 1
            phase_payload = {
                **phase_info,
                "phase_gate_granularity": context["granularity"],
                "allowed_phase_ids": [int(item) for item in allowed_ids],
                "allowed_phases": [
                    context["phase_names"][int(item)] for item in allowed_ids
                ],
                "next_phase_window": (
                    None if next_window_value is None else int(next_window_value)
                ),
                "allowed_demo_states": int(allowed_mask.sum().item()),
                "finite_after_gate": int(finite_after),
            }
            self._record_task_phase_info(phase_payload)
        return gated

    def _soft_phase_gate_distances(
        self,
        distances,
        phase_info,
        distance_config,
        context,
        record=True,
    ):
        finite_mask = torch.isfinite(distances)
        finite_count = int(finite_mask.sum().item())
        if finite_count <= 0:
            return distances

        phase_tensor = context["phase_tensor"]
        phase_step_tensor = context["phase_step_tensor"]
        prefix = str(context.get("prefix", self.task))
        current_phase_id = int(context["phase_id"])
        phase_delta = phase_tensor - current_phase_id
        penalty_units = torch.zeros_like(distances)

        current_penalty = float(
            distance_config.get(f"{prefix}_phase_gate_soft_current_penalty", 0.0)
        )
        next_penalty = float(
            distance_config.get(f"{prefix}_phase_gate_soft_next_penalty", 0.0)
        )
        next_step_penalty = float(
            distance_config.get(f"{prefix}_phase_gate_soft_next_step_penalty", 0.75)
        )
        previous_penalty = float(
            distance_config.get(f"{prefix}_phase_gate_soft_previous_penalty", 1.5)
        )
        skip_penalty = float(
            distance_config.get(f"{prefix}_phase_gate_soft_skip_penalty", 4.0)
        )
        allow_next = bool(distance_config.get(f"{prefix}_phase_gate_allow_next", True))

        current_mask = phase_delta == 0
        next_mask = phase_delta == 1
        previous_mask = phase_delta < 0
        skip_future_mask = phase_delta > 1

        if current_penalty != 0.0:
            penalty_units[current_mask] = current_penalty

        if allow_next:
            step_values = phase_step_tensor.to(
                device=distances.device,
                dtype=distances.dtype,
            )
            next_window = distance_config.get(
                f"{prefix}_phase_gate_soft_next_window",
                distance_config.get(f"{prefix}_phase_gate_next_window", None),
            )
            if next_window is None:
                next_progress = torch.zeros_like(distances)
            else:
                next_window_value = max(1.0, float(next_window))
                next_max_progress = float(
                    distance_config.get(f"{prefix}_phase_gate_soft_next_max_progress", 3.0)
                )
                next_progress = torch.clamp(
                    step_values / next_window_value,
                    min=0.0,
                    max=max(0.0, next_max_progress),
                )
            penalty_units[next_mask] = (
                next_penalty + next_step_penalty * next_progress[next_mask]
            )
        else:
            penalty_units[next_mask] = skip_penalty

        if previous_penalty > 0.0:
            penalty_units[previous_mask] = (
                previous_penalty
                * torch.abs(phase_delta[previous_mask]).to(
                    device=distances.device,
                    dtype=distances.dtype,
                )
            )
        if skip_penalty > 0.0:
            penalty_units[skip_future_mask] = (
                skip_penalty
                * torch.abs(phase_delta[skip_future_mask]).to(
                    device=distances.device,
                    dtype=distances.dtype,
                )
            )

        top_n = min(
            max(2, int(distance_config.get(f"{prefix}_phase_gate_soft_scale_top_n", 64))),
            finite_count,
        )
        finite_distances = distances[finite_mask]
        if top_n >= 2:
            top_distances = torch.topk(
                finite_distances,
                k=top_n,
                largest=False,
            ).values
            local_scale = torch.median(top_distances) - top_distances[0]
            penalty_scale = float(local_scale.item())
        else:
            penalty_scale = 0.0
        penalty_scale = max(
            penalty_scale,
            float(distance_config.get(f"{prefix}_phase_gate_soft_min_penalty", 0.005)),
        )

        softened = distances + penalty_units * penalty_scale
        hard_boundary = bool(
            distance_config.get(f"{prefix}_phase_gate_soft_hard_boundary", False)
        )
        boundary_mask = current_mask | (next_mask if allow_next else torch.zeros_like(next_mask))
        if hard_boundary:
            softened[finite_mask & (~boundary_mask)] = float("inf")
        if record:
            key = f"{context['granularity']}:{context['phase']}"
            self._phase_gate_counts[key] = self._phase_gate_counts.get(key, 0) + 1
            penalized_mask = finite_mask & (penalty_units > 0.0)
            allowed_ids = self._allowed_phase_ids_for_context(
                context,
                distance_config,
                phase_info=phase_info,
            )
            self._record_task_phase_info({
                **phase_info,
                "phase_gate_granularity": context["granularity"],
                "phase_gate_mode": "soft",
                "allowed_phase_ids": [int(item) for item in allowed_ids],
                "allowed_phases": [
                    context["phase_names"][int(item)] for item in allowed_ids
                ],
                "phase_soft_penalty_scale": float(penalty_scale),
                "phase_soft_penalized_states": int(penalized_mask.sum().item()),
                "phase_soft_hard_boundary": bool(hard_boundary),
                "phase_soft_previous_states": int((finite_mask & previous_mask).sum().item()),
                "phase_soft_next_states": int((finite_mask & next_mask).sum().item()),
                "phase_soft_skip_future_states": int(
                    (finite_mask & skip_future_mask).sum().item()
                ),
                "finite_after_gate": int(finite_count),
            })
        return softened

    def _lift_start_preference_distances(self, distances, phase_info, distance_config, record=True):
        profile = self.task_profile
        if profile is None:
            return distances
        prefix = profile.name
        if (
            not bool(distance_config.get(f"{prefix}_lift_start_preference", False))
            or not bool(phase_info.get("lift_start_ready", False))
        ):
            return distances
        if not hasattr(self, "demo_lift_start_window_tensor"):
            return distances

        finite_mask = torch.isfinite(distances)
        preferred_mask = self.demo_lift_start_window_tensor & finite_mask
        if not bool(preferred_mask.any().item()):
            return distances

        hard_gate = bool(
            distance_config.get(f"{prefix}_lift_start_preference_hard_gate", False)
        )
        boost = float(distance_config.get(f"{prefix}_lift_start_preference_boost", 0.7))
        penalty = float(distance_config.get(f"{prefix}_lift_start_preference_penalty", 0.0))
        if not hard_gate and boost >= 1.0 and penalty <= 0.0:
            return distances

        preferred = distances.clone()
        if hard_gate:
            preferred[finite_mask & (~self.demo_lift_start_window_tensor)] = float("inf")
        if 0.0 < boost < 1.0:
            preferred[preferred_mask] = preferred[preferred_mask] * boost
        penalize = finite_mask & (~self.demo_lift_start_window_tensor)
        if penalty > 0.0 and not hard_gate:
            preferred[penalize] = preferred[penalize] + penalty
        if record:
            key = phase_info["phase"]
            self._lift_start_preference_counts[key] = (
                self._lift_start_preference_counts.get(key, 0) + 1
            )
            phase_payload = {
                **self._current_task_phase_info(),
                "lift_start_preference": True,
                "lift_start_hard_gate": bool(hard_gate),
                "lift_start_boost": float(boost),
                "lift_start_penalty": float(penalty),
                "lift_start_window_states": int(self.demo_lift_start_window_tensor.sum().item()),
                "lift_start_finite_window_states": int(preferred_mask.sum().item()),
                "lift_start_penalized_states": int(penalize.sum().item()),
            }
            self._record_task_phase_info(phase_payload)
        return preferred
