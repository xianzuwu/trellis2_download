import ast
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


MESH_EXTENSIONS = {
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
}


def get_file_hash(file):
    sha256 = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()


def get_string_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def resolve_manifest_path(root, manifest=None, default_name="manifest.csv"):
    if manifest is not None:
        return os.path.expanduser(manifest)
    path = os.path.join(root, default_name)
    if os.path.exists(path):
        return path
    return None


def read_manifest(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            records = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return pd.DataFrame.from_records(records)
        data = json.load(open(path, "r"))
        if isinstance(data, dict):
            records = []
            for key, value in data.items():
                if isinstance(value, dict):
                    records.append({"file_identifier": key, **value})
                else:
                    records.append({"file_identifier": key, "value": value})
            return pd.DataFrame.from_records(records)
        return pd.DataFrame.from_records(data)
    return pd.read_csv(path)


def normalize_mesh_manifest(metadata, root=None, default_dataset_name=None):
    metadata = metadata.copy()
    if "local_path" in metadata.columns and "file_identifier" not in metadata.columns:
        metadata["file_identifier"] = metadata["local_path"]
    if "file_identifier" not in metadata.columns:
        raise ValueError("Manifest must contain either a file_identifier or local_path column")
    if "sha256" not in metadata.columns:
        metadata["sha256"] = None
    if "record_id" not in metadata.columns:
        metadata["record_id"] = metadata["file_identifier"].apply(get_string_hash)
    if root is not None:
        for idx, row in metadata.iterrows():
            if pd.notna(row.get("sha256")):
                continue
            path = None
            if pd.notna(row.get("local_path")):
                path = row["local_path"]
                if not os.path.isabs(path):
                    path = os.path.join(root, path)
            elif pd.notna(row.get("source_path")):
                path = os.path.expanduser(row["source_path"])
            if path is not None and os.path.exists(path):
                metadata.at[idx, "sha256"] = get_file_hash(path)
    metadata["sha256"] = [
        sha256 if pd.notna(sha256) else get_string_hash(file_identifier)
        for sha256, file_identifier in zip(metadata["sha256"], metadata["file_identifier"])
    ]
    if "dataset" not in metadata.columns and default_dataset_name is not None:
        metadata["dataset"] = default_dataset_name
    return metadata


def build_metadata_from_local_files(root, raw_dir="raw", default_dataset_name=None):
    base = os.path.join(root, raw_dir)
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"No metadata manifest was found and {base} does not exist. "
            "Place assets under the raw directory or pass --manifest."
        )

    records = []
    for current_root, _, files in os.walk(base):
        for name in files:
            if Path(name).suffix.lower() not in MESH_EXTENSIONS:
                continue
            path = os.path.join(current_root, name)
            rel_path = os.path.relpath(path, root)
            records.append(
                {
                    "sha256": get_file_hash(path),
                    "file_identifier": rel_path,
                    "local_path": rel_path,
                }
            )

    metadata = pd.DataFrame.from_records(records)
    if len(metadata) == 0:
        raise FileNotFoundError(f"No supported mesh files found under {base}")
    if default_dataset_name is not None:
        metadata["dataset"] = default_dataset_name
    return metadata


def parse_list_value(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def select_path_by_resolution(paths, resolution=None):
    paths = parse_list_value(paths)
    if len(paths) == 0:
        return None
    if resolution is None:
        return paths[0]

    resolution = str(resolution).lower()
    resolution_map = {
        "1k": "1024",
        "2k": "2048",
        "4k": "4096",
        "8k": "8192",
        "1024": "1024",
        "2048": "2048",
        "4096": "4096",
        "8192": "8192",
    }
    target = resolution_map.get(resolution, resolution)
    for path in paths:
        stem = Path(path).stem
        if stem.endswith(f"_{target}"):
            return path
    return paths[0]


def ensure_sha256(path, expected_sha256=None):
    sha256 = get_file_hash(path)
    if (
        expected_sha256 is not None
        and not pd.isna(expected_sha256)
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256))
        and expected_sha256 != sha256
    ):
        raise ValueError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {sha256}")
    return sha256


def copy_local_asset(src_path, output_dir, expected_sha256=None):
    src_path = os.path.expanduser(src_path)
    sha256 = ensure_sha256(src_path, expected_sha256)
    suffix = Path(src_path).suffix
    dst_rel = os.path.join("raw", "local", f"{sha256}{suffix}")
    dst_path = os.path.join(output_dir, dst_rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return sha256, dst_rel


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects", no_file=False):
    records = []
    max_workers = max_workers or os.cpu_count()

    def worker(metadatum):
        sha256 = metadatum.get("sha256", "unknown")
        try:
            if no_file:
                record = func(None, metadatum)
            else:
                local_path = metadatum["local_path"]
                file_path = local_path if os.path.isabs(local_path) else os.path.join(output_dir, local_path)
                record = func(file_path, metadatum)
            if record is not None:
                records.append(record)
        except Exception as e:
            print(f"Error processing object {sha256}: {e}")
        finally:
            pbar.update()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc=desc) as pbar:
            for metadatum in metadata.to_dict("records"):
                executor.submit(worker, metadatum)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")

    return pd.DataFrame.from_records(records)


def is_remote_identifier(identifier):
    if not isinstance(identifier, str):
        return False
    scheme = urlparse(identifier).scheme
    return scheme in {"http", "https"}
