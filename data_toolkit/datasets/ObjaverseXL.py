import argparse
import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile
import zipfile

import objaverse.xl as oxl
import pandas as pd
from tqdm import tqdm

from datasets.common import get_file_hash

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, batch_size=None, **kwargs):
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    batch_size = batch_size or int(os.environ.get("TRELLIS2_OBJAVERSEXL_BATCH_SIZE", "5000"))

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    if len(annotations) == 0:
        return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

    num_batches = math.ceil(len(annotations) / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(len(annotations), (batch_idx + 1) * batch_size)
        batch = annotations.iloc[start:end]
        try:
            file_paths = oxl.download_objects(
                batch,
                download_dir=os.path.join(output_dir, "raw"),
                save_repo_format="zip",
            )
        except BaseException as e:
            print(f"Error downloading ObjaverseXL batch {batch_idx + 1}/{num_batches}: {e}")
            continue

        for file_identifier, path in file_paths.items():
            try:
                sha256 = metadata.loc[file_identifier, "sha256"]
            except Exception:
                continue
            downloaded[sha256] = os.path.relpath(path, output_dir)

    # Recover files that objaverse wrote before a batch-level exception was raised.
    for _, metadatum in metadata.reset_index().iterrows():
        sha256 = metadatum["sha256"]
        if sha256 in downloaded:
            continue
        for candidate in os.path.join(output_dir, "raw", "**", f"{sha256}.*"), os.path.join(output_dir, "raw", "**", f"{sha256}"):
            matches = glob.glob(candidate, recursive=True)
            if matches:
                downloaded[sha256] = os.path.relpath(matches[0], output_dir)
                break

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', no_file=False):
    records = []
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            
            def worker(metadatum):
                try:
                    sha256 = metadatum['sha256']
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum['local_path']
                        if local_path.startswith('raw/github/repos/'):
                            path_parts = local_path.split('/')
                            file_name = os.path.join(*path_parts[5:])
                            zip_file = os.path.join(output_dir, *path_parts[:5])
                            import tempfile, zipfile
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                    zip_ref.extractall(tmp_dir)
                                file = os.path.join(tmp_dir, file_name)
                                record = func(file, metadatum) 
                        else:
                            file = os.path.join(output_dir, local_path)
                            record = func(file, metadatum) 

                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {metadatum.get('sha256', 'unknown')}: {e}")
                    pbar.update()
            
            for metadatum in metadata.to_dict('records'):
                executor.submit(worker, metadatum)
            
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
        
    return pd.DataFrame.from_records(records)
