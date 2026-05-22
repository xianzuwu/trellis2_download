import os
import time

import huggingface_hub
import pandas as pd

from datasets.common import (
    build_metadata_from_local_files,
    copy_local_asset,
    ensure_sha256,
    foreach_instance,
    get_file_hash,
    normalize_mesh_manifest,
    read_manifest,
    resolve_manifest_path,
    select_path_by_resolution,
)


TEXVERSE_REPOS = {
    "1k": "YiboZhang2001/TexVerse-1K",
    "1024": "YiboZhang2001/TexVerse-1K",
    "2k": "YiboZhang2001/TexVerse",
    "2048": "YiboZhang2001/TexVerse",
    "4k": "YiboZhang2001/TexVerse",
    "4096": "YiboZhang2001/TexVerse",
    "8k": "YiboZhang2001/TexVerse",
    "8192": "YiboZhang2001/TexVerse",
}


def add_args(parser):
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional metadata manifest. Defaults to <root>/manifest.csv if present.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="2k",
        choices=["1k", "2k", "4k", "8k", "1024", "2048", "4096", "8192"],
        help="Preferred TexVerse GLB texture resolution.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Local TexVerse metadata.json path. If omitted, it is downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Override the Hugging Face dataset repo used for selected TexVerse resolution.",
    )


def _repo_id(resolution, repo_id=None):
    return repo_id or TEXVERSE_REPOS[resolution.lower()]


def _repo_id_for_path(path, repo_id=None):
    if repo_id is not None:
        return repo_id
    if isinstance(path, str) and "/glbs_1k/" in path:
        return "YiboZhang2001/TexVerse-1K"
    return "YiboZhang2001/TexVerse"


def _load_texverse_metadata(metadata_file, resolution, repo_id, root):
    if metadata_file is None:
        os.makedirs(os.path.join(root, "hf_metadata"), exist_ok=True)
        metadata_file = huggingface_hub.hf_hub_download(
            repo_id=repo_id or "YiboZhang2001/TexVerse",
            filename="metadata.json",
            repo_type="dataset",
            local_dir=os.path.join(root, "hf_metadata"),
        )
    data = pd.read_json(metadata_file, orient="index")
    data = data.reset_index().rename(columns={"index": "file_identifier"})
    data["source_id"] = data["file_identifier"]
    data["selected_glb_path"] = data["glb_paths"].apply(lambda paths: select_path_by_resolution(paths, resolution))
    data = data[data["selected_glb_path"].notna()].copy()
    data["selected_repo_id"] = data["selected_glb_path"].apply(lambda path: _repo_id_for_path(path, repo_id))
    data["sha256"] = data["source_id"]
    data["dataset"] = "TexVerse"
    return data


def get_metadata(root, manifest=None, resolution="2k", metadata_file=None, repo_id=None, **kwargs):
    manifest_path = resolve_manifest_path(root, manifest)
    if manifest_path is not None:
        return normalize_mesh_manifest(read_manifest(manifest_path), root=root, default_dataset_name="TexVerse")

    try:
        return _load_texverse_metadata(metadata_file, resolution, repo_id, root)
    except Exception as e:
        print("\033[93m")
        print(f"Failed to load TexVerse metadata from Hugging Face: {e}")
        print("Falling back to local files under <root>/raw. Pass --manifest for a custom subset.")
        print("\033[0m")
        return build_metadata_from_local_files(root, default_dataset_name="TexVerse")


def _download_hf_file(repo_id, filename, output_dir):
    path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.join(output_dir, "raw"),
    )
    return os.path.relpath(path, output_dir)


def _write_batch_records(records, output_dir, batch_index):
    if len(records) == 0:
        return
    part_dir = os.path.join(output_dir, "raw", "new_records")
    os.makedirs(part_dir, exist_ok=True)
    part_path = os.path.join(part_dir, f"part_texverse_{int(time.time())}_{batch_index}.csv")
    pd.DataFrame.from_records(records, columns=["sha256", "local_path"]).to_csv(part_path, index=False)


def download(metadata, output_dir, resolution="2k", repo_id=None, record_batch_size=100, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    records = []
    pending_records = []

    if len(metadata) == 0:
        return pd.DataFrame.from_records(records, columns=["sha256", "local_path"])

    for idx, metadatum in enumerate(metadata.to_dict("records"), start=1):
        try:
            expected_sha256 = metadatum.get("sha256")
            if "local_path" in metadatum and pd.notna(metadatum["local_path"]):
                local_path = metadatum["local_path"]
                path = local_path if os.path.isabs(local_path) else os.path.join(output_dir, local_path)
                sha256 = ensure_sha256(path, expected_sha256)
                record = {"sha256": expected_sha256 or sha256, "local_path": local_path}
                records.append(record)
                pending_records.append(record)
                continue

            if "source_path" in metadatum and pd.notna(metadatum["source_path"]):
                sha256, local_path = copy_local_asset(metadatum["source_path"], output_dir, expected_sha256)
                record = {"sha256": expected_sha256 or sha256, "local_path": local_path}
                records.append(record)
                pending_records.append(record)
                continue

            glb_path = metadatum.get("selected_glb_path")
            if glb_path is None or pd.isna(glb_path):
                glb_path = select_path_by_resolution(metadatum.get("glb_paths"), resolution)
            if glb_path is None:
                print(f"Skipping {metadatum.get('file_identifier', 'unknown')}: no GLB path")
                continue

            download_repo_id = metadatum.get("selected_repo_id")
            if download_repo_id is None or pd.isna(download_repo_id):
                download_repo_id = _repo_id_for_path(glb_path, repo_id)
            local_path = _download_hf_file(download_repo_id, glb_path, output_dir)
            record = {"sha256": expected_sha256, "local_path": local_path}
            records.append(record)
            pending_records.append(record)
        except Exception as e:
            print(f"Error downloading {metadatum.get('file_identifier', 'unknown')}: {e}")

        if len(pending_records) >= record_batch_size:
            _write_batch_records(pending_records, output_dir, idx)
            pending_records = []

    _write_batch_records(pending_records, output_dir, "final")
    return pd.DataFrame.from_records(records, columns=["sha256", "local_path"])
