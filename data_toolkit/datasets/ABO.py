import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from datasets.common import foreach_instance, get_file_hash


def add_args(parser):
    pass


def get_metadata(**kwargs):
    return pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ABO.csv")


def download(metadata, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    archive_path = os.path.join(output_dir, "raw", "abo-3dmodels.tar")

    if not os.path.exists(archive_path):
        try:
            os.system(
                f"wget -O {archive_path} "
                "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
            )
        except Exception:
            print("\033[93m")
            print("Error downloading ABO dataset. Please check your internet connection and try again.")
            print(f"Or manually download abo-3dmodels.tar and place it in {output_dir}/raw")
            print("Visit https://amazon-berkeley-objects.s3.amazonaws.com/index.html for more information")
            print("\033[0m")
            raise FileNotFoundError("Error downloading ABO dataset")

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    with tarfile.open(archive_path) as tar:
        with ThreadPoolExecutor(max_workers=1) as executor, tqdm(total=len(metadata), desc="Extracting") as pbar:
            def worker(instance):
                try:
                    tar.extract(f"3dmodels/original/{instance}", path=os.path.join(output_dir, "raw"))
                    sha256 = get_file_hash(os.path.join(output_dir, "raw/3dmodels/original", instance))
                    return sha256
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
            downloaded[sha256] = os.path.join("raw/3dmodels/original", file_identifier)
        else:
            print(f"Error downloading {file_identifier}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])
