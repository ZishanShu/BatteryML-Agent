# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import time
import math
import requests

from tqdm import tqdm
from pathlib import Path


DOWNLOAD_LINKS = {
    'MATR': [
        ('https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download',
         'MATR_batch_20170512.mat'),
        ('https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download',
         'MATR_batch_20170630.mat'),
        ('https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download',
         'MATR_batch_20180412.mat'),
        ('https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download',
         'MATR_batch_20190124.mat')
    ],
    'HUST': [
        ('https://data.mendeley.com/public-files/datasets/nsc7hnsg4s/'
         'files/5ca0ac3e-d598-4d07-8dcb-879aa047e98b/file_downloaded',
         'hust_data.zip'),
    ],
    'CALCE': [
        ('https://web.calce.umd.edu/batteries/data/CS2_33.zip', 'CS2_33.zip'),
        ('https://web.calce.umd.edu/batteries/data/CS2_34.zip', 'CS2_34.zip'),
        ('https://web.calce.umd.edu/batteries/data/CS2_35.zip', 'CS2_35.zip'),
        ('https://web.calce.umd.edu/batteries/data/CS2_36.zip', 'CS2_36.zip'),
        ('https://web.calce.umd.edu/batteries/data/CS2_37.zip', 'CS2_37.zip'),
        ('https://web.calce.umd.edu/batteries/data/CS2_38.zip', 'CS2_38.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_16.zip', 'CX2_16.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_33.zip', 'CX2_33.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_35.zip', 'CX2_35.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_34.zip', 'CX2_34.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_36.zip', 'CX2_36.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_37.zip', 'CX2_37.zip'),
        ('https://web.calce.umd.edu/batteries/data/CX2_38.zip', 'CX2_38.zip'),
    ],
    'RWTH': [
        ('https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip',
         'RWTH.zip')
    ]
}


def is_download_complete(filename):
    """Return whether a downloaded dataset file is usable.

    MATR files are HDF5 containers. A partially transferred file can be
    several gigabytes and still have a non-zero size, so checking only the
    byte count incorrectly marks it as complete. Opening the HDF5 header is
    enough to catch truncated files without reading the dataset contents.
    """
    filename = Path(filename)
    if not filename.is_file() or filename.stat().st_size <= 0:
        return False
    if filename.suffix.lower() != '.mat' or not filename.name.startswith('MATR_batch_'):
        return True
    try:
        import h5py

        with h5py.File(filename, 'r') as handle:
            return 'batch' in handle
    except (OSError, ValueError):
        return False


def download_file(url,
                  filename,
                  update_interval=500,
                  chunk_size=4096,
                  total_length=None,
                  force=False):
    def memory2str(mem):
        sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        power = int(math.log(mem, 1024))
        size = sizes[power]
        for _ in range(power):
            mem /= 1024
        if power > 0:
            return f'{mem:.2f}{size}'
        else:
            return f'{mem}{size}'
    if not filename.parent.exists():
        filename.parent.mkdir()
    if filename.exists() and not force:
        # A previous interrupted request can leave an empty placeholder
        # behind.  Treat that as missing so a later Download action retries
        # the file; the old existence-only check incorrectly skipped it and
        # left MATR permanently incomplete on Windows.
        if is_download_complete(filename):
            print(f'[INFO] {filename} already exists. Skip it.')
            return
        if filename.is_file():
            filename.unlink()
    # Download into a sidecar file and publish atomically.  A connection
    # failure must never leave a zero-byte (or partial) dataset file that
    # looks complete to the next run.
    temp_filename = filename.with_name(filename.name + '.part')
    if temp_filename.exists():
        temp_filename.unlink()
    try:
        response = requests.get(url, stream=True, verify=False)
        if total_length is None:
            total_length = response.headers.get('content-length')
        if response.status_code != 200:
            raise ValueError(
                f'Network error: {response.status_code}! URL: {url}')
        with open(temp_filename, 'wb') as f:
            if total_length is None:
                f.write(response.content)
            else:
                downloaded, total_length = 0, int(total_length)
                total_size = memory2str(total_length)
                bar_format = (
                    f'Downloading {filename}'
                    '|{percentage:3.0f}%|{bar:20}|{desc}'
                    '[{elapsed}<{remaining}{postfix}]')
                if update_interval * chunk_size * 100 >= total_length:
                    update_interval = 1
                with tqdm(total=total_length, bar_format=bar_format) as bar:
                    counter = 0
                    now_time, now_size = time.time(), downloaded
                    for data in response.iter_content(chunk_size=chunk_size):
                        f.write(data)
                        downloaded += len(data)
                        counter += 1
                        bar.update(len(data))
                        if counter % update_interval == 0:
                            elapsed = time.time() - now_time
                            runtime_downloaded = downloaded - now_size
                            now_time, now_size = time.time(), downloaded

                            cur_size = memory2str(downloaded)
                            if elapsed > 0:
                                speed_size = memory2str(
                                    runtime_downloaded / elapsed)
                            else:
                                speed_size = '0B/s'
                            bar.set_description(f'{cur_size}/{total_size}')
                            bar.set_postfix_str(f'{speed_size}/s')

                            counter = 0
        temp_filename.replace(filename)
    except (requests.exceptions.RequestException, OSError, ValueError):
        if temp_filename.exists():
            temp_filename.unlink()
        raise
