# GPI_robomimic

This repository contains a minimal reproduction of PCA-state GPI policies for
the RoboMimic Can, Square, and Lift tasks.

The rollout policy used for the reported numbers lives in
`src/gpi_repro/policy/`, with the command-line entry point in
`src/gpi_repro/policy/evaluate.py`. The implementation is split across
`config.py`, `state.py`, `pca.py`, `phase.py`, `action.py`, `gpi_policy.py`,
`rollout.py`, and `cli.py`. Task configs live in `configs/`.
PCA caches are generated from the RoboMimic datasets on first use and written
to `artifacts/pca_state/`.

Datasets, generated PCA caches, rollout videos, and result logs are recreated
by the commands below and are not stored in the repository.

## Setup

Install the package and rollout dependencies:

```bash
conda create -n gpi-robomimic python=3.11
conda activate gpi-robomimic
pip install -e ".[rollout]"
```

The `rollout` extra installs RoboMimic from the official `v0.5.0` GitHub tag,
since that version is not published on PyPI. The package versions used for
reproduction are pinned in `pyproject.toml`.

Can seed-level results are sensitive to the low-dimensional object observation
layout, so use a consistent robosuite / RoboMimic / MuJoCo environment when
comparing exact seed lists.

You need the low-dimensional RoboMimic datasets. The runner can download the
exact files used here from the Diffusion Policy training-data archive:

```bash
scripts/download_data.sh
```

This downloads
`https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip`,
extracts only the three hdf5 files below, verifies their sha256 checksums, and
removes the zip unless `--keep-archive` is passed. By default the runner expects:

```text
data/
  robomimic/datasets/can/mh/low_dim_abs.hdf5
  robomimic/datasets/square/mh/low_dim_abs.hdf5
  robomimic/datasets/lift/mh/low_dim.hdf5
```

Use `--data-root` or per-task overrides (`--can-dataset`, `--square-dataset`,
`--lift-dataset`) if your files are elsewhere.

## Re-run Rollouts

Run the default open-source reproduction suite, seeds `0..30`:

```bash
scripts/reproduce.sh --download-data --seeds 0..30
```

Run one task:

```bash
scripts/reproduce.sh --task can --download-data --seeds 0..30
```

Save one rollout video per seed:

```bash
scripts/reproduce.sh \
  --download-data \
  --seeds 0..30 \
  --workers 10 \
  --mujoco-gl egl \
  --save-videos \
  --output-root results/reproduce_0_30_videos
```

Each task directory writes `per_seed_results.csv`, and the output root writes
`per_seed_results_all_tasks.csv`. These files contain each seed's success flag,
worker directory, and video path.

Print the exact worker commands without launching robosuite:

```bash
scripts/reproduce.sh --dry-run --seeds 0..30
```

Useful runtime flags:

- `--workers N`: split seeds across parallel Python processes.
- `--download-data`: fetch missing RoboMimic hdf5 files before running.
- `--device cuda:0`: device passed to the policy script.
- `--mujoco-gl egl`: set `MUJOCO_GL` for headless rendering systems.
- `--rebuild-cache`: force PCA cache regeneration from datasets.
- `--extra-policy-args --horizon 500`: append arguments directly to the policy.

## Reproduced Results

The `results/` directory is ignored by git and is not needed for reproduction.
The runner writes fresh `summary.json`, `checkpoints_every_10.csv`,
`per_seed_results.csv`, `final_summary.csv`, and combined checkpoint / per-seed
files under the selected `--output-root`.

The reported reproduction numbers use seeds `0..30`:

These counts were reproduced with the default dependency pins in
`pyproject.toml` on:

```text
Python 3.11.15
torch 2.11.0+cu128
torchvision 0.26.0+cu128
robomimic 0.5.0
robosuite 1.5.1
mujoco 3.3.5
numpy 1.26.4
h5py 3.16.0
imageio 2.33.1
opencv-python 4.11.0.86
mink 0.0.13
```

| Task | Success |
| --- | ---: |
| Can | `30/31 = 96.7742%` |
| Square | `27/31 = 87.0968%` |
| Lift | `31/31 = 100.0000%` |
