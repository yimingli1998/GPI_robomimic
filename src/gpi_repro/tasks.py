from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.environ.get("GPI_REPRO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            (candidate / "configs").is_dir()
            and (candidate / "src" / "gpi_repro" / "policy" / "evaluate.py").exists()
        ):
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
DEFAULT_POLICY_SCRIPT = PROJECT_ROOT / "src" / "gpi_repro" / "policy" / "evaluate.py"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "reproduce_0_30"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "artifacts" / "pca_state"

TASKS = ("can", "square", "lift")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_relpath: Path
    config: Path
    cache_name: str
    args: tuple[str, ...]

    def dataset_path(self, data_root: Path) -> Path:
        return data_root / self.dataset_relpath

    def cache_path(self, cache_root: Path) -> Path:
        return cache_root / self.cache_name


COMMON_PCA_ARGS = (
    "--pca-state-variance-power",
    "0.5",
)

FUTURE_PROGRESS_CAN = (
    "--future-progress-mode",
    "candidate",
    "--future-progress-ori-weight",
    "0.0",
)

FUTURE_PROGRESS_SQUARE = (
    "--future-progress-mode",
    "candidate",
    "--future-progress-top-n",
    "48",
    "--future-progress-horizon",
    "24",
    "--future-progress-success-weight",
    "0.5130534231703093",
    "--future-progress-distance-ratio",
    "2.270860613618215",
    "--future-progress-distance-margin",
    "0.055830456883200055",
)

SQUARE_LOCAL_RESIDUAL = (
    "--local-residual",
    "--local-residual-min-phase",
    "pre_insert",
    "--local-residual-target-offset",
    "5",
    "--local-residual-pos-alpha",
    "0.49640198032761507",
    "--local-residual-ori-alpha",
    "0.3104971888220224",
    "--local-residual-max-pos",
    "0.006206310556809735",
    "--local-residual-max-rot",
    "0.1275809118364365",
)

TASK_SPECS = {
    "can": TaskSpec(
        name="can",
        dataset_relpath=Path("robomimic/datasets/can/mh/low_dim_abs.hdf5"),
        config=PROJECT_ROOT / "configs" / "can.json",
        cache_name="can.pt",
        args=(
            "--fresh-env-per-seed",
            "--distance-config",
            str(PROJECT_ROOT / "configs" / "can.json"),
            *COMMON_PCA_ARGS,
            *FUTURE_PROGRESS_CAN,
            "--flow-field",
            "--flow-attraction",
            "1.5",
            "--flow-target-offset",
            "8",
        ),
    ),
    "square": TaskSpec(
        name="square",
        dataset_relpath=Path("robomimic/datasets/square/mh/low_dim_abs.hdf5"),
        config=PROJECT_ROOT / "configs" / "square.json",
        cache_name="square.pt",
        args=(
            "--fresh-env-per-seed",
            "--distance-config",
            str(PROJECT_ROOT / "configs" / "square.json"),
            *COMMON_PCA_ARGS,
            "--pca-state-components",
            "26",
            *FUTURE_PROGRESS_SQUARE,
            *SQUARE_LOCAL_RESIDUAL,
            "--pick-success-lift-delta",
            "0.028003168002070367",
            "--pick-success-eef-object-max",
            "0.0877638076697665",
            "--progress-guard",
            "--progress-guard-min-position",
            "220",
            "--progress-guard-closed-only",
        ),
    ),
    "lift": TaskSpec(
        name="lift",
        dataset_relpath=Path("robomimic/datasets/lift/mh/low_dim.hdf5"),
        config=PROJECT_ROOT / "configs" / "lift.json",
        cache_name="lift.pt",
        args=(
            "--action-horizon",
            "1",
            "--action-retarget",
            "none",
            "--env-control-mode",
            "delta",
            "--fresh-env-per-seed",
            "--no-pick-success-demo-calibrated",
            "--pick-success-lift-delta",
            "0.02",
            "--pick-success-eef-object-max",
            "0.05",
            "--distance-config",
            str(PROJECT_ROOT / "configs" / "lift.json"),
            *COMMON_PCA_ARGS,
            "--pca-state-components",
            "5",
        ),
    ),
}


def task_names() -> tuple[str, ...]:
    return TASKS


def get_task(name: str) -> TaskSpec:
    try:
        return TASK_SPECS[name]
    except KeyError as exc:
        known = ", ".join(TASKS)
        raise ValueError(f"unknown task {name!r}; expected one of: {known}") from exc
