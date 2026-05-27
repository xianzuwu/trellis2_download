import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import huggingface_hub
import pandas as pd
from tqdm import tqdm

from datasets.common import foreach_instance, get_file_hash


def add_args(parser):
    pass


def get_metadata(**kwargs):
    return pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv")


def download(metadata, output_dir, max_workers=None, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    max_workers = max_workers or int(os.environ.get("TRELLIS2_HSSD_MAX_WORKERS", "8"))

    try:
        huggingface_hub.whoami()
    except Exception:
        print("\033[93m")
        print("Haven't logged in to the Hugging Face Hub.")
        print("Visit https://huggingface.co/settings/tokens to get a token.")
        print("\033[0m")
        huggingface_hub.login()

    try:
        huggingface_hub.hf_hub_download(repo_id="hssd/hssd-models", filename="README.md", repo_type="dataset")
    except Exception:
        print("\033[93m")
        print("Error downloading HSSD dataset.")
        print("Check if you have access to https://huggingface.co/datasets/hssd/hssd-models")
        print("\033[0m")

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc="Downloading") as pbar:
        def worker(instance):
            try:
                huggingface_hub.hf_hub_download(
                    repo_id="hssd/hssd-models",
                    filename=instance,
                    repo_type="dataset",
                    local_dir=os.path.join(output_dir, "raw"),
                )
                return get_file_hash(os.path.join(output_dir, "raw", instance))
            except BaseException as e:
                print(f"Error downloading for {instance}: {e}")
                return None
            finally:
                pbar.update()

        futures = {executor.submit(worker, instance): instance for instance in metadata.index}
        for future in as_completed(futures):
            file_identifier = futures[future]
            try:
                sha256 = future.result()
            except BaseException as e:
                print(f"Error downloading for {file_identifier}: {e}")
                continue

            if sha256 is None:
                continue
            if sha256 == metadata.loc[file_identifier, "sha256"]:
                downloaded[sha256] = os.path.join("raw", file_identifier)
            else:
                print(f"Error downloading {file_identifier}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])
