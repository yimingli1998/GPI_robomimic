import pathlib

import numpy as np
import torch

from .config import *
from .state import normalize_quat


class PCAMixin:
    def _state_to_vec(self, state):
        pieces = []
        for key in OBS_KEYS:
            if key in QUATERNION_FEATURES:
                value = normalize_quat(state[key])
            else:
                value = (state[key] - self.norm_stats[key]["mean"]) / self.norm_stats[key]["std"]
            value = value * np.sqrt(self.distance_weights.get(key, 1.0))
            pieces.append(value.reshape(-1))
        return np.concatenate(pieces).astype(np.float32)

    def _build_tensors(self):
        state_vecs = np.asarray([self._state_to_vec(state) for state in self.state_components])
        self.state_vecs_np = state_vecs.astype(np.float32)
        self.state_tensor = torch.as_tensor(state_vecs, dtype=torch.float32, device=self.device)
        self.demo_index_tensor = torch.as_tensor(self.demo_indices, dtype=torch.long, device=self.device)
        self.demo_position_tensor = torch.as_tensor(self.flat_demo_positions, dtype=torch.long, device=self.device)

    def _build_pca_state_space(self):
        if self.knn_space != "pca_state":
            return
        if (
            self.pca_state_cache is not None
            and self.pca_state_cache.exists()
            and not self.pca_state_rebuild_cache
        ):
            self._load_pca_state_cache(self.pca_state_cache)
            return
        x = np.asarray(self.state_vecs_np, dtype=np.float64)
        n_components = max(1, min(int(self.pca_state_components), x.shape[1]))
        mean = x.mean(axis=0)
        centered = x - mean.reshape(1, -1)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components].astype(np.float32)
        explained_variance = (
            (singular_values[:n_components] ** 2) / max(1, x.shape[0] - 1)
        ).astype(np.float32)
        embedding = centered @ components.T.astype(np.float64)
        if self.pca_state_variance_power != 0.0:
            embedding = embedding / np.power(
                explained_variance.reshape(1, -1) + 1e-8,
                0.5 * self.pca_state_variance_power,
            )
        self.pca_state_mean = mean.astype(np.float32)
        self.pca_state_components_matrix = components
        self.pca_state_explained_variance = explained_variance
        self.pca_state_demo_tensor = torch.as_tensor(
            embedding.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        if self.pca_state_cache is not None:
            self._save_pca_state_cache(self.pca_state_cache)
        total_var = float(np.sum(np.var(x, axis=0)))
        explained = float(np.sum(explained_variance) / (total_var + 1e-12))
        print(
            "Built PCA state KNN space: "
            f"N={x.shape[0]}, input_dim={x.shape[1]}, components={n_components}, "
            f"variance_power={self.pca_state_variance_power:.4f}, "
            f"explained={explained:.6f}"
        )

    def _load_pca_state_cache(self, path):
        cache = torch.load(path, map_location=self.device, weights_only=False)
        cached_demo_embedding = torch.as_tensor(
            cache["demo_embedding"],
            dtype=torch.float32,
            device=self.device,
        )
        if cached_demo_embedding.shape[0] != len(self.state_components):
            raise ValueError(
                f"PCA state cache demo count mismatch: cache has {cached_demo_embedding.shape[0]}, "
                f"dataset has {len(self.state_components)}"
            )
        self.pca_state_mean = np.asarray(cache["mean"], dtype=np.float32)
        self.pca_state_components_matrix = np.asarray(
            cache["components"],
            dtype=np.float32,
        )
        self.pca_state_explained_variance = np.asarray(
            cache["explained_variance"],
            dtype=np.float32,
        )
        cache_power = float(cache.get("variance_power", 1.0 if cache.get("whiten", False) else 0.0))
        if abs(cache_power - self.pca_state_variance_power) <= 1e-8:
            demo_embedding = cached_demo_embedding
        else:
            centered = (
                np.asarray(self.state_vecs_np, dtype=np.float32)
                - self.pca_state_mean.reshape(1, -1)
            )
            embedding = centered @ self.pca_state_components_matrix.T
            if self.pca_state_variance_power != 0.0:
                embedding = embedding / np.power(
                    self.pca_state_explained_variance.reshape(1, -1) + 1e-8,
                    0.5 * self.pca_state_variance_power,
                )
            demo_embedding = torch.as_tensor(
                embedding.astype(np.float32),
                dtype=torch.float32,
                device=self.device,
            )
        self.pca_state_demo_tensor = demo_embedding
        print(
            "Loaded PCA state KNN cache: "
            f"{path}, N={demo_embedding.shape[0]}, "
            f"components={demo_embedding.shape[1]}, "
            f"variance_power={self.pca_state_variance_power:.4f}"
        )

    def _save_pca_state_cache(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "task": self.task,
            "dataset_path": str(self.dataset_path),
            "distance_config": self.distance_config,
            "components_requested": self.pca_state_components,
            "variance_power": self.pca_state_variance_power,
            "mean": np.asarray(self.pca_state_mean, dtype=np.float32),
            "components": np.asarray(
                self.pca_state_components_matrix,
                dtype=np.float32,
            ),
            "explained_variance": np.asarray(
                self.pca_state_explained_variance,
                dtype=np.float32,
            ),
            "demo_embedding": self.pca_state_demo_tensor.detach().cpu(),
        }
        torch.save(payload, path)
        print(f"Saved PCA state KNN cache: {path}")

    def _pca_state_distances(self, current_state):
        if self.pca_state_demo_tensor is None:
            raise RuntimeError("PCA state space has not been built")
        x = self._state_to_vec(current_state).astype(np.float32)
        centered = x - self.pca_state_mean
        query = centered @ self.pca_state_components_matrix.T
        if self.pca_state_variance_power != 0.0:
            query = query / np.power(
                self.pca_state_explained_variance + 1e-8,
                0.5 * self.pca_state_variance_power,
            )
        query_tensor = torch.as_tensor(
            query,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return torch.sum((self.pca_state_demo_tensor - query_tensor) ** 2, dim=1)
