import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from datasets.common import foreach_instance, get_file_hash


def add_args(parser):
    pass


def get_metadata(**kwargs):
    return pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv")


def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    archive_path = os.path.join(output_dir, "raw", "toys4k_blend_files.zip")

    if not os.path.exists(archive_path):
        print("\033[93m")
        print("Toys4k has to be downloaded manually.")
        print(f"Please download toys4k_blend_files.zip and place it in {output_dir}/raw")
        print("Visit https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k for more information")
        print("\033[0m")
        raise FileNotFoundError("toys4k_blend_files.zip not found")

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    with zipfile.ZipFile(archive_path) as zip_ref:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(metadata), desc="Extracting") as pbar:
            def worker(instance):
                try:
                    zip_ref.extract(os.path.join("toys4k_blend_files", instance), os.path.join(output_dir, "raw"))
                    return get_file_hash(os.path.join(output_dir, "raw/toys4k_blend_files", instance))
                except Exception as e:
                    print(f"Error extracting for {instance}: {e}")
                    return None
                finally:
                    pbar.update()

            sha256s = executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    for file_identifier, sha256 in zip(metadata.index, sha256s):
        if sha256 is None:
            continue
        if sha256 == metadata.loc[file_identifier, "sha256"]:
            downloaded[sha256] = os.path.join("raw/toys4k_blend_files", file_identifier)
        else:
            print(f"Error downloading {file_identifier}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])
