import os

import pandas as pd

from datasets.common import (
    build_metadata_from_local_files,
    copy_local_asset,
    ensure_sha256,
    foreach_instance,
    get_file_hash,
    is_remote_identifier,
    normalize_mesh_manifest,
    read_manifest,
    resolve_manifest_path,
)


def add_args(parser):
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "CSV/JSON/JSONL manifest for the picked Sketchfab assets. "
            "Defaults to <root>/manifest.csv if present."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sketchfab",
        choices=["sketchfab", "local"],
        help="Download from Objaverse-XL Sketchfab annotations or use local files listed in the manifest.",
    )


def get_metadata(root, manifest=None, **kwargs):
    manifest_path = resolve_manifest_path(root, manifest)
    if manifest_path is not None:
        return normalize_mesh_manifest(read_manifest(manifest_path), root=root, default_dataset_name="SketchfabPicked")

    print("\033[93m")
    print("SketchfabPicked metadata is not published in the TRELLIS.2 repo yet.")
    print("Pass --manifest with file_identifier values from Objaverse-XL, or place mesh files under <root>/raw.")
    print("\033[0m")
    return build_metadata_from_local_files(root, default_dataset_name="SketchfabPicked")


def _download_objaverse(metadata, output_dir):
    import objaverse.xl as oxl

    annotations = oxl.get_annotations()
    ids = metadata["file_identifier"].astype(str).values
    annotations = annotations[annotations["file_identifier"].isin(ids)]
    if "sha256" in metadata.columns:
        annotations = annotations[
            annotations["sha256"].isin(metadata["sha256"].dropna().astype(str).values)
            | annotations["file_identifier"].isin(ids)
        ]

    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
    )

    annotation_sha256 = annotations.set_index("file_identifier")["sha256"].to_dict()
    metadata = metadata.set_index("file_identifier")
    records = []
    for file_identifier, path in file_paths.items():
        expected_sha256 = metadata.loc[file_identifier].get("sha256")
        if expected_sha256 is None or pd.isna(expected_sha256):
            expected_sha256 = annotation_sha256.get(file_identifier)
        if expected_sha256 is None or pd.isna(expected_sha256):
            expected_sha256 = get_file_hash(path)
        records.append({"sha256": expected_sha256, "local_path": os.path.relpath(path, output_dir)})
    return records


def download(metadata, output_dir, source="sketchfab", **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

    if len(metadata) == 0:
        return pd.DataFrame.from_records([], columns=["sha256", "local_path"])

    if "local_path" in metadata.columns:
        records = []
        for metadatum in metadata.to_dict("records"):
            local_path = metadatum.get("local_path")
            if pd.isna(local_path):
                continue
            path = local_path if os.path.isabs(local_path) else os.path.join(output_dir, local_path)
            sha256 = metadatum.get("sha256")
            if sha256 is None or pd.isna(sha256):
                sha256 = get_file_hash(path)
            else:
                ensure_sha256(path, sha256)
            records.append({"sha256": sha256, "local_path": local_path})
        if len(records) > 0:
            return pd.DataFrame.from_records(records, columns=["sha256", "local_path"])

    if "source_path" in metadata.columns:
        records = []
        for metadatum in metadata.to_dict("records"):
            source_path = metadatum.get("source_path")
            if pd.isna(source_path):
                continue
            sha256, local_path = copy_local_asset(source_path, output_dir, metadatum.get("sha256"))
            records.append({"sha256": metadatum.get("sha256") or sha256, "local_path": local_path})
        if len(records) > 0:
            return pd.DataFrame.from_records(records, columns=["sha256", "local_path"])

    if source == "local":
        raise FileNotFoundError("No local_path or source_path entries found in SketchfabPicked manifest")

    if any(is_remote_identifier(x) for x in metadata["file_identifier"].astype(str).values):
        raise ValueError(
            "Objaverse-XL download expects file_identifier values from its annotations, not direct URLs. "
            "Use source_path/local_path for local files."
        )

    records = _download_objaverse(metadata, output_dir)
    return pd.DataFrame.from_records(records, columns=["sha256", "local_path"])
