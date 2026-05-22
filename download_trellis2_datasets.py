#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DATA_TOOLKIT = REPO_ROOT / "data_toolkit"
sys.path.insert(0, str(DATA_TOOLKIT))


SUPPORTED_DATASETS = ["ObjaverseXL", "ABO", "HSSD", "TexVerse", "SketchfabPicked", "Toys4k"]
DEFAULT_DATASETS = ["ObjaverseXL", "ABO", "HSSD", "TexVerse", "SketchfabPicked", "Toys4k"]
DATASET_PROFILES = {
    "train": ["ObjaverseXL", "ABO", "HSSD", "TexVerse"],
    "test": ["SketchfabPicked", "Toys4k"],
    "all": DEFAULT_DATASETS,
}


@dataclass
class DatasetJob:
    label: str
    module_name: str
    root: Path
    get_kwargs: dict
    download_kwargs: dict


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_dataset_module(module_name: str):
    return importlib.import_module(f"datasets.{module_name}")


def load_or_build_metadata(job: DatasetJob):
    job.root.mkdir(parents=True, exist_ok=True)
    metadata_path = job.root / "metadata.csv"
    if metadata_path.exists():
        return pd.read_csv(metadata_path)

    module = load_dataset_module(job.module_name)
    metadata = module.get_metadata(root=str(job.root), **job.get_kwargs)
    metadata.to_csv(metadata_path, index=False)
    return metadata


def merge_part_files(records_dir: Path):
    records_dir.mkdir(parents=True, exist_ok=True)
    new_dir = records_dir / "new_records"
    merged_dir = records_dir / "merged_records"
    new_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    part_files = sorted(new_dir.glob("part_*.csv"))
    metadata_path = records_dir / "metadata.csv"
    existing = read_csv_if_exists(metadata_path)

    merged = None
    if existing is not None and len(existing) > 0 and "sha256" in existing.columns:
        merged = existing.set_index("sha256")

    changed = False
    for part_file in part_files:
        try:
            part = pd.read_csv(part_file)
        except Exception as e:
            print(f"Failed to read {part_file}: {e}")
            continue
        if len(part) == 0 or "sha256" not in part.columns:
            shutil.move(str(part_file), str(merged_dir / f"{time.strftime('%Y%m%d%H%M%S')}_{part_file.name}"))
            changed = True
            continue
        part = part.set_index("sha256")
        if merged is None:
            merged = part
        else:
            merged = part.combine_first(merged)
        shutil.move(str(part_file), str(merged_dir / f"{time.strftime('%Y%m%d%H%M%S')}_{part_file.name}"))
        changed = True

    if merged is None:
        if existing is not None:
            return existing
        return None

    merged = merged.reset_index()
    merged.to_csv(metadata_path, index=False)
    return merged


def overlay_metadata(base: pd.DataFrame | None, overlay: pd.DataFrame | None):
    if overlay is None or len(overlay) == 0:
        return base
    if base is None or len(base) == 0:
        return overlay.copy()
    if "sha256" not in base.columns or "sha256" not in overlay.columns:
        raise ValueError("metadata frames must contain sha256")
    return overlay.set_index("sha256").combine_first(base.set_index("sha256")).reset_index()


def sync_metadata(job: DatasetJob, master: pd.DataFrame | None):
    raw_dir = job.root / "raw"
    raw_meta = merge_part_files(raw_dir)
    if raw_meta is None:
        raw_meta = read_csv_if_exists(raw_dir / "metadata.csv")
    if raw_meta is not None:
        raw_meta.to_csv(raw_dir / "metadata.csv", index=False)
        master = overlay_metadata(master, raw_meta)
        if master is not None:
            job.root.mkdir(parents=True, exist_ok=True)
            master.to_csv(job.root / "metadata.csv", index=False)
    return master


def collect_existing_local_paths(root: Path):
    raw_dir = root / "raw"
    if not raw_dir.exists():
        return set()

    suffixes = {
        ".blend",
        ".fbx",
        ".glb",
        ".gltf",
        ".obj",
        ".ply",
        ".stl",
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
        ".zip",
        ".tar",
    }
    paths = set()
    for path in raw_dir.rglob("*"):
        if not path.is_file() or ".cache" in path.parts or path.suffix.lower() not in suffixes:
            continue
        paths.add(path.relative_to(root).as_posix())
    return paths


def recover_existing_downloads(job: DatasetJob, metadata: pd.DataFrame | None):
    if metadata is None or len(metadata) == 0 or "sha256" not in metadata.columns:
        return None

    existing_paths = collect_existing_local_paths(job.root)
    if len(existing_paths) == 0:
        return None

    records = []
    for metadatum in metadata.to_dict("records"):
        local_path = metadatum.get("local_path")
        if pd.notna(local_path) and str(local_path) in existing_paths:
            records.append({"sha256": metadatum["sha256"], "local_path": str(local_path)})
            continue

        candidates = []
        for column in ("selected_glb_path", "file_identifier"):
            value = metadatum.get(column)
            if pd.notna(value):
                candidates.append((Path("raw") / str(value)).as_posix())

        for candidate in candidates:
            if candidate in existing_paths:
                records.append({"sha256": metadatum["sha256"], "local_path": candidate})
                break

    if len(records) == 0:
        return None

    recovered = pd.DataFrame.from_records(records).drop_duplicates("sha256")
    part_dir = job.root / "raw" / "new_records"
    part_dir.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(part_dir / f"part_recovered_{time.strftime('%Y%m%d%H%M%S')}.csv", index=False)
    print(f"  recovered existing files: {len(recovered)}")
    return recovered


def file_exists(root: Path, local_path) -> bool:
    if pd.isna(local_path):
        return False
    path = Path(str(local_path))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def find_missing(metadata: pd.DataFrame, root: Path):
    if metadata is None or len(metadata) == 0:
        return metadata, metadata
    if "local_path" not in metadata.columns:
        return metadata.copy(), metadata.copy()
    present_mask = metadata["local_path"].apply(lambda p: file_exists(root, p))
    present = metadata[present_mask].copy()
    missing = metadata[~present_mask].copy()
    return present, missing


def prepare_download_subset(missing: pd.DataFrame):
    if missing is None or len(missing) == 0:
        return missing
    subset = missing.copy()
    if "local_path" in subset.columns:
        subset = subset.drop(columns=["local_path"])
    return subset


def run_job(job: DatasetJob, retries: int, sleep_seconds: int):
    print(f"\n==> {job.label} @ {job.root}")
    module = load_dataset_module(job.module_name)
    metadata = load_or_build_metadata(job)
    metadata = sync_metadata(job, metadata)
    recover_existing_downloads(job, metadata)
    metadata = sync_metadata(job, metadata)

    attempts = 0
    while attempts <= retries:
        metadata = sync_metadata(job, metadata)
        present, missing = find_missing(metadata, job.root)
        print(f"  present: {len(present)}  missing: {len(missing)}")
        if len(missing) == 0:
            print("  done")
            return True

        attempts += 1
        if attempts > retries:
            break

        subset = prepare_download_subset(missing)
        if subset is None or len(subset) == 0:
            print("  nothing to download")
            break

        part_path = job.root / "raw" / "new_records" / f"part_{attempts}.csv"
        part_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded = module.download(subset, output_dir=str(job.root), **job.download_kwargs)
        except Exception as e:
            print(f"  download failed on attempt {attempts}: {e}")
            downloaded = None

        if downloaded is not None and len(downloaded) > 0:
            downloaded.to_csv(part_path, index=False)
            metadata = sync_metadata(job, metadata)
        else:
            print(f"  no new records on attempt {attempts}")

        if attempts < retries:
            time.sleep(max(1, sleep_seconds) * attempts)

    metadata = sync_metadata(job, metadata)
    _, missing = find_missing(metadata, job.root)
    if len(missing) > 0:
        print(f"  unresolved: {len(missing)}")
        if "file_identifier" in missing.columns:
            preview = missing["file_identifier"].astype(str).head(10).tolist()
            print(f"  first missing: {', '.join(preview)}")
        return False
    print("  done")
    return True


def build_jobs(args):
    jobs = []
    base_root = Path(args.root).resolve()
    requested = DATASET_PROFILES.get(args.datasets, parse_csv_list(args.datasets))

    for dataset in requested:
        if dataset == "ObjaverseXL":
            for source in parse_csv_list(args.objaversexl_sources):
                jobs.append(
                    DatasetJob(
                        label=f"ObjaverseXL[{source}]",
                        module_name="ObjaverseXL",
                        root=base_root / f"ObjaverseXL_{source}",
                        get_kwargs={"source": source},
                        download_kwargs={"source": source},
                    )
                )
        elif dataset == "ABO":
            jobs.append(
                DatasetJob(
                    label="ABO",
                    module_name="ABO",
                    root=base_root / "ABO",
                    get_kwargs={},
                    download_kwargs={},
                )
            )
        elif dataset == "HSSD":
            jobs.append(
                DatasetJob(
                    label="HSSD",
                    module_name="HSSD",
                    root=base_root / "HSSD",
                    get_kwargs={},
                    download_kwargs={},
                )
            )
        elif dataset == "TexVerse":
            for resolution in parse_csv_list(args.texverse_resolution):
                jobs.append(
                    DatasetJob(
                        label=f"TexVerse[{resolution}]",
                        module_name="TexVerse",
                        root=base_root / f"TexVerse_{resolution}",
                        get_kwargs={
                            "resolution": resolution,
                            "metadata_file": args.texverse_metadata_file,
                            "repo_id": args.texverse_repo_id,
                            "manifest": args.texverse_manifest,
                        },
                        download_kwargs={
                            "resolution": resolution,
                            "repo_id": args.texverse_repo_id,
                        },
                    )
                )
        elif dataset == "SketchfabPicked":
            jobs.append(
                DatasetJob(
                    label="SketchfabPicked",
                    module_name="SketchfabPicked",
                    root=base_root / "SketchfabPicked",
                    get_kwargs={
                        "manifest": args.sketchfab_picked_manifest,
                    },
                    download_kwargs={"source": "sketchfab"},
                )
            )
        elif dataset == "Toys4k":
            jobs.append(
                DatasetJob(
                    label="Toys4k",
                    module_name="Toys4k",
                    root=base_root / "Toys4k",
                    get_kwargs={},
                    download_kwargs={},
                )
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Download and resume TRELLIS.2 datasets using metadata-aware retries.")
    parser.add_argument("--root", type=str, default="datasets", help="Base directory for dataset roots")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated list of datasets, or profile train/test/all. "
            "Options: " + ", ".join(SUPPORTED_DATASETS)
        ),
    )
    parser.add_argument(
        "--objaversexl-sources",
        type=str,
        default="sketchfab,github",
        help="Comma-separated ObjaverseXL sources to download",
    )
    parser.add_argument(
        "--texverse-resolution",
        type=str,
        default="2k",
        help="Comma-separated TexVerse resolutions to download",
    )
    parser.add_argument(
        "--texverse-metadata-file",
        type=str,
        default=None,
        help="Optional local TexVerse metadata.json path",
    )
    parser.add_argument(
        "--texverse-repo-id",
        type=str,
        default=None,
        help="Optional TexVerse repo override",
    )
    parser.add_argument(
        "--texverse-manifest",
        type=str,
        default=None,
        help="Optional TexVerse manifest for a custom subset",
    )
    parser.add_argument(
        "--sketchfab-picked-manifest",
        type=str,
        default=None,
        help="Optional manifest for SketchfabPicked when the official picked list is not available",
    )
    parser.add_argument("--retries", type=int, default=5, help="Number of retry rounds for missing files")
    parser.add_argument("--sleep-seconds", type=int, default=5, help="Base sleep between retry rounds")
    args = parser.parse_args()

    jobs = build_jobs(args)
    ok = True
    for job in jobs:
        try:
            ok = run_job(job, retries=args.retries, sleep_seconds=args.sleep_seconds) and ok
        except Exception as e:
            ok = False
            print(f"\n!! {job.label} failed: {e}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
