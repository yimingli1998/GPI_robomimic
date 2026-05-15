from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


ARCHIVE_URL = "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip"
ARCHIVE_SHA256 = "70e8c297b5928988145c39852d55900f22d4ea4afe0ecac89bb8fc3b3fcb0e53"
ARCHIVE_NAME = "robomimic_lowdim.zip"


@dataclass(frozen=True)
class DatasetFile:
    member: str
    relpath: Path
    sha256: str


REQUIRED_DATASETS = (
    DatasetFile(
        member="robomimic/datasets/can/mh/low_dim_abs.hdf5",
        relpath=Path("robomimic/datasets/can/mh/low_dim_abs.hdf5"),
        sha256="9b5dd4e672f5e30bbabf8e31c58230d316d4bc820186ed98c5c768462a573878",
    ),
    DatasetFile(
        member="robomimic/datasets/square/mh/low_dim_abs.hdf5",
        relpath=Path("robomimic/datasets/square/mh/low_dim_abs.hdf5"),
        sha256="405eec2d47f0741964b271a05af427a68e62283f60b1f23c889ab4a3967336a2",
    ),
    DatasetFile(
        member="robomimic/datasets/lift/mh/low_dim.hdf5",
        relpath=Path("robomimic/datasets/lift/mh/low_dim.hdf5"),
        sha256="7a2b9f40017a75b649b074f9f3c49ab92da3e9fd03794ef28b8c7687c0649904",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_or_invalid(data_root: Path) -> list[DatasetFile]:
    needed = []
    for item in REQUIRED_DATASETS:
        path = data_root / item.relpath
        if not path.exists() or sha256_file(path) != item.sha256:
            needed.append(item)
    return needed


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    last_reported_pct = -5

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_reported_pct
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        pct = 100.0 * downloaded / total_size
        pct_floor = int(pct // 5) * 5
        if pct_floor >= last_reported_pct + 5 or downloaded == total_size:
            last_reported_pct = pct_floor
            print(f"[download] {downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB ({pct:5.1f}%)")

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, tmp, reporthook=report)
    os.replace(tmp, output)


def extract_datasets(archive: Path, data_root: Path, items: list[DatasetFile]) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for item in items:
            if item.member not in names:
                raise FileNotFoundError(f"{item.member} not found in {archive}")
            target = data_root / item.relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            print(f"[extract] {item.member} -> {target}")
            with zf.open(item.member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            actual = sha256_file(tmp)
            if actual != item.sha256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"sha256 mismatch for {target}: expected {item.sha256}, got {actual}")
            os.replace(tmp, target)


def ensure_datasets(
    data_root: Path,
    *,
    url: str = ARCHIVE_URL,
    archive_path: Path | None = None,
    force: bool = False,
    keep_archive: bool = False,
) -> None:
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    needed = list(REQUIRED_DATASETS) if force else missing_or_invalid(data_root)
    if not needed:
        print(f"[data] RoboMimic datasets are ready under {data_root}")
        return

    archive = archive_path.expanduser().resolve() if archive_path else data_root / "downloads" / ARCHIVE_NAME
    if archive_path is None and (force or not archive.exists()):
        download_file(url, archive)
    elif not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    if archive_path is None and url == ARCHIVE_URL:
        actual = sha256_file(archive)
        if actual != ARCHIVE_SHA256:
            raise RuntimeError(f"sha256 mismatch for {archive}: expected {ARCHIVE_SHA256}, got {actual}")

    extract_datasets(archive, data_root, needed)
    if archive_path is None and not keep_archive:
        archive.unlink(missing_ok=True)
        download_dir = archive.parent
        if download_dir.exists() and not any(download_dir.iterdir()):
            download_dir.rmdir()
    print(f"[data] RoboMimic datasets are ready under {data_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the RoboMimic datasets used by this reproduction.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--archive", default=None, help="Use an existing robomimic_lowdim.zip instead of downloading.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if checked files already exist.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded zip under <data-root>/downloads.")
    args = parser.parse_args(argv)

    ensure_datasets(
        Path(args.data_root),
        url=args.url,
        archive_path=None if args.archive is None else Path(args.archive),
        force=args.force,
        keep_archive=args.keep_archive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
