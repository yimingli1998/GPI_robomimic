from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .data_download import ARCHIVE_URL, ensure_datasets
from .tasks import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POLICY_SCRIPT,
    DEFAULT_PYTHON,
    DEFAULT_THRESHOLDS,
    TASKS,
    TaskSpec,
    get_task,
)


def parse_seeds(seed_arg: str) -> list[int]:
    if ".." in seed_arg:
        start_s, end_s = seed_arg.split("..", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"invalid seed range {seed_arg!r}")
        return list(range(start, end + 1))
    return [int(part.strip()) for part in seed_arg.split(",") if part.strip()]


def format_seed_span(seeds: list[int]) -> str:
    if not seeds:
        return "empty"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}..{seeds[-1]}"
    return ",".join(str(seed) for seed in seeds)


def split_seeds(seeds: list[int], workers: int) -> list[list[int]]:
    workers = max(1, min(workers, len(seeds)))
    chunk_size = math.ceil(len(seeds) / workers)
    return [seeds[i : i + chunk_size] for i in range(0, len(seeds), chunk_size)]


def task_dataset(args: argparse.Namespace, task: TaskSpec) -> Path:
    override = getattr(args, f"{task.name}_dataset")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(args.data_root).expanduser().resolve() / task.dataset_relpath).resolve()


def command_for(
    *,
    python: Path,
    policy_script: Path,
    task: TaskSpec,
    dataset: Path,
    output_dir: Path,
    seeds: str,
    device: str,
    horizon: int,
    cache_path: Path,
    extra_policy_args: list[str],
    video_dir: Path | None = None,
) -> list[str]:
    cmd = [
        str(python),
        str(policy_script),
        "--dataset",
        str(dataset),
        "--task",
        task.name,
        "--output-dir",
        str(output_dir),
        "--seeds",
        seeds,
        "--device",
        device,
        "--horizon",
        str(horizon),
        "--pca-state-cache",
        str(cache_path),
        *task.args,
        *extra_policy_args,
    ]
    if video_dir is not None:
        cmd.extend(["--video-dir", str(video_dir)])
    return cmd


def ensure_cache(
    *,
    args: argparse.Namespace,
    task: TaskSpec,
    dataset: Path,
    cache_path: Path,
) -> None:
    if args.skip_cache_build:
        return
    if cache_path.exists() and not args.rebuild_cache:
        return
    if not dataset.exists():
        raise FileNotFoundError(
            f"{task.name} dataset not found: {dataset}\n"
            "Pass --download-data, --data-root, or --<task>-dataset to point at the RoboMimic hdf5 file."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.output_root).expanduser().resolve() / f"_cache_build_{task.name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = command_for(
        python=Path(args.python).expanduser(),
        policy_script=Path(args.policy_script).expanduser().resolve(),
        task=task,
        dataset=dataset,
        output_dir=build_dir,
        seeds="0",
        device=args.device,
        horizon=args.horizon,
        cache_path=cache_path,
        extra_policy_args=[*args.extra_policy_args, "--build-pca-state-cache"],
        video_dir=None,
    )
    print(f"[cache] building {task.name}: {cache_path}")
    subprocess.run(cmd, check=True, env=rollout_env(args))


def rollout_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    src = str((Path(__file__).resolve().parents[1]).resolve())
    env["PYTHONPATH"] = src
    if args.mujoco_gl:
        env["MUJOCO_GL"] = args.mujoco_gl
    return env


def load_worker_results(worker_dir: Path) -> list[dict[str, Any]]:
    log_path = worker_dir / "eval_log.json"
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for result in payload.get("results", []):
        item = dict(result)
        item["_worker_dir"] = str(worker_dir)
        rows.append(item)
    return rows


def summarize(task: TaskSpec, seeds: list[int], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {int(result["seed"]): result for result in results}
    ordered = [by_seed[seed] for seed in seeds if seed in by_seed]
    successes = [int(result["seed"]) for result in ordered if result.get("success")]
    failures = [int(result["seed"]) for result in ordered if not result.get("success")]
    pick_successes = [int(result["seed"]) for result in ordered if result.get("pick_success")]
    pick_failures = [int(result["seed"]) for result in ordered if not result.get("pick_success")]
    total = len(ordered)
    expected = task.expected_success_by_total.get(total)
    return {
        "task": task.name,
        "profile": "pca_state",
        "protocol": "single_pass_robosuite_rollout",
        "seeds": format_seed_span(seeds),
        "num_success": len(successes),
        "num_total": total,
        "success_rate": len(successes) / total if total else 0.0,
        "successful_seeds": successes,
        "failed_seeds": failures,
        "num_pick_success": len(pick_successes),
        "pick_success_rate": len(pick_successes) / total if total else 0.0,
        "pick_successful_seeds": pick_successes,
        "pick_failed_seeds": pick_failures,
        "expected_success": expected,
        "matches_reference_count": None if expected is None else len(successes) == expected,
        "missing_seeds": [seed for seed in seeds if seed not in by_seed],
    }


def checkpoints(task_name: str, seeds: list[int], successful: set[int], every: int) -> list[dict[str, Any]]:
    rows = []
    end_indices = list(range(0, len(seeds), every))
    if end_indices[-1] != len(seeds) - 1:
        end_indices.append(len(seeds) - 1)
    for end_idx in end_indices:
        chunk = seeds[: end_idx + 1]
        success = sum(1 for seed in chunk if seed in successful)
        rows.append(
            {
                "task": task_name,
                "n": len(chunk),
                "start": seeds[0],
                "end": chunk[-1],
                "success": success,
                "fail": len(chunk) - success,
                "success_rate": success / len(chunk),
            }
        )
    return rows


def per_seed_rows(
    *,
    task: TaskSpec,
    seeds: list[int],
    results: list[dict[str, Any]],
    save_videos: bool,
    video_root: Path | None,
) -> list[dict[str, Any]]:
    by_seed = {int(result["seed"]): result for result in results}
    rows = []
    for seed in seeds:
        result = by_seed.get(seed)
        worker_dir = Path(result.get("_worker_dir", "")) if result is not None else None
        worker_name = "" if worker_dir is None else worker_dir.name
        video_path = ""
        if save_videos and video_root is not None and worker_name:
            video_path = str(video_root / task.name / worker_name / f"seed_{seed:03d}.mp4")
        rows.append(
            {
                "task": task.name,
                "seed": seed,
                "success": "" if result is None else bool(result.get("success")),
                "pick_success": "" if result is None else bool(result.get("pick_success")),
                "task_success": "" if result is None else bool(result.get("task_success")),
                "horizon": "" if result is None else int(result.get("horizon", 0)),
                "return": "" if result is None else float(result.get("return", 0.0)),
                "worker": worker_name,
                "video_path": video_path,
            }
        )
    return rows


def write_summary(
    task_dir: Path,
    summary: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    with (task_dir / "checkpoints_every_10.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "n", "start", "end", "success", "fail", "success_rate"],
        )
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    with (task_dir / "per_seed_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "seed",
                "success",
                "pick_success",
                "task_success",
                "horizon",
                "return",
                "worker",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_rows)


def write_combined(
    output_root: Path,
    summaries: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> None:
    with (output_root / "final_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "success",
                "total",
                "fail",
                "success_rate",
                "pick_success",
                "pick_success_rate",
                "failed_seeds",
            ],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "task": item["task"],
                    "success": item["num_success"],
                    "total": item["num_total"],
                    "fail": item["num_total"] - item["num_success"],
                    "success_rate": item["success_rate"],
                    "pick_success": item["num_pick_success"],
                    "pick_success_rate": item["pick_success_rate"],
                    "failed_seeds": " ".join(str(seed) for seed in item["failed_seeds"]),
                }
            )
    with (output_root / "checkpoints_every_10_all_tasks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "n", "start", "end", "success", "fail", "success_rate"],
        )
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    with (output_root / "per_seed_results_all_tasks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "seed",
                "success",
                "pick_success",
                "task_success",
                "horizon",
                "return",
                "worker",
                "video_path",
            ],
        )
        writer.writeheader()
        writer.writerows(seed_rows)


def run_task(
    args: argparse.Namespace,
    task: TaskSpec,
    seeds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = Path(args.output_root).expanduser().resolve()
    task_dir = output_root / f"{task.name}_{seeds[0]}_{seeds[-1]}"
    worker_root = task_dir / "workers"
    dataset = task_dataset(args, task)
    cache_path = (Path(args.cache_root).expanduser().resolve() / task.cache_name).resolve()
    policy_script = Path(args.policy_script).expanduser().resolve()
    python = Path(args.python).expanduser()

    if not args.dry_run:
        if not policy_script.exists():
            raise FileNotFoundError(f"policy script not found: {policy_script}")
        if not dataset.exists():
            raise FileNotFoundError(
                f"{task.name} dataset not found: {dataset}\n"
                "Pass --download-data, --data-root, or --<task>-dataset to point at the RoboMimic hdf5 file."
            )
        ensure_cache(args=args, task=task, dataset=dataset, cache_path=cache_path)

    chunks = split_seeds(seeds, args.workers)
    commands = []
    for idx, chunk in enumerate(chunks):
        worker_dir = worker_root / f"worker_{idx:02d}_{chunk[0]}_{chunk[-1]}"
        video_dir = None
        if args.save_videos:
            video_dir = (
                Path(args.video_root).expanduser().resolve()
                / task.name
                / f"worker_{idx:02d}_{chunk[0]}_{chunk[-1]}"
            )
        cmd = command_for(
            python=python,
            policy_script=policy_script,
            task=task,
            dataset=dataset,
            output_dir=worker_dir,
            seeds=format_seed_span(chunk),
            device=args.device,
            horizon=args.horizon,
            cache_path=cache_path,
            extra_policy_args=args.extra_policy_args,
            video_dir=video_dir,
        )
        commands.append((worker_dir, cmd))

    if args.dry_run:
        print(f"\n# {task.name}")
        print(f"# dataset: {dataset}")
        print(f"# cache:   {cache_path}")
        for _, cmd in commands:
            print(" ".join(cmd))
        empty_summary = summarize(task, seeds, [])
        return empty_summary, [], []

    print(f"[run] {task.name}: {len(seeds)} seeds across {len(commands)} worker(s)")
    processes = []
    for worker_dir, cmd in commands:
        worker_dir.mkdir(parents=True, exist_ok=True)
        log_file = (worker_dir / "stdout.log").open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=rollout_env(args))
        processes.append((process, log_file, worker_dir))

    failed = []
    for process, log_file, worker_dir in processes:
        code = process.wait()
        log_file.close()
        if code != 0:
            failed.append((worker_dir, code))
    if failed:
        details = "\n".join(f"{path} exited with {code}" for path, code in failed)
        raise RuntimeError(f"{task.name} worker failure(s):\n{details}")

    results: list[dict[str, Any]] = []
    for worker_dir, _ in commands:
        results.extend(load_worker_results(worker_dir))
    summary = summarize(task, seeds, results)
    checkpoint_rows = checkpoints(task.name, seeds, set(summary["successful_seeds"]), args.checkpoint_every)
    seed_rows = per_seed_rows(
        task=task,
        seeds=seeds,
        results=results,
        save_videos=bool(args.save_videos),
        video_root=None if args.video_root is None else Path(args.video_root).expanduser().resolve(),
    )
    write_summary(task_dir, summary, checkpoint_rows, seed_rows)
    print(
        f"[done] {task.name}: {summary['num_success']}/{summary['num_total']} "
        f"= {summary['success_rate'] * 100:.4f}%"
    )
    return summary, checkpoint_rows, seed_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-run PCA-state GPI RoboMimic rollouts.")
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--seeds", default="0..30")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--download-data", action="store_true", help="Download required RoboMimic hdf5 files if missing.")
    parser.add_argument("--force-download-data", action="store_true", help="Re-download / re-extract RoboMimic hdf5 files.")
    parser.add_argument("--data-url", default=ARCHIVE_URL, help="Zip URL used by --download-data.")
    parser.add_argument("--data-archive", default=None, help="Existing robomimic_lowdim.zip to extract instead of downloading.")
    parser.add_argument("--keep-data-archive", action="store_true", help="Keep the downloaded zip after extraction.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--policy-script", default=str(DEFAULT_POLICY_SCRIPT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--can-dataset", default=None)
    parser.add_argument("--square-dataset", default=None)
    parser.add_argument("--lift-dataset", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--mujoco-gl", default=None, help="Optional MUJOCO_GL override, e.g. egl.")
    parser.add_argument("--skip-cache-build", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument(
        "--video-root",
        default=None,
        help="Directory for rollout videos. Defaults to <output-root>/videos when --save-videos is set.",
    )
    parser.add_argument(
        "--extra-policy-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended verbatim to the vendored policy script.",
    )
    args = parser.parse_args(argv)
    if args.save_videos and args.video_root is None:
        args.video_root = str(Path(args.output_root).expanduser().resolve() / "videos")

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("no seeds selected")
    selected = TASKS if args.task == "all" else (args.task,)
    output_root = Path(args.output_root).expanduser().resolve()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    if args.download_data and not args.dry_run:
        ensure_datasets(
            Path(args.data_root),
            url=args.data_url,
            archive_path=None if args.data_archive is None else Path(args.data_archive),
            force=args.force_download_data,
            keep_archive=args.keep_data_archive,
        )

    summaries: list[dict[str, Any]] = []
    all_checkpoints: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []
    for name in selected:
        summary, rows, seed_rows = run_task(args, get_task(name), seeds)
        summaries.append(summary)
        all_checkpoints.extend(rows)
        all_seed_rows.extend(seed_rows)

    if not args.dry_run:
        write_combined(output_root, summaries, all_checkpoints, all_seed_rows)
        print(f"[write] {output_root}")
        for item in summaries:
            expected = item.get("expected_success")
            if expected is not None:
                status = "MATCH" if item["num_success"] == expected else "DIFF"
                print(f"[{status}] {item['task']} reference {expected}/{item['num_total']}")
            else:
                threshold = DEFAULT_THRESHOLDS.get(item["task"])
                if threshold is None:
                    continue
                status = "PASS" if item["success_rate"] >= threshold else "FAIL"
                print(f"[{status}] {item['task']} threshold {threshold * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
