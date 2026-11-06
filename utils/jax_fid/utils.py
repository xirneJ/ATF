import os
import random
import tempfile
import time

import requests
from tqdm import tqdm

# Environment variables that let you provide the pretrained InceptionV3 weights
# locally (the default Dropbox URL is not reachable on some clusters):
#   JAX_FID_CKPT_PATH: full path to the weights file
#       e.g. ~/qiangchen/data/fid/jax_fid/inception_v3_weights_fid.pickle
#   JAX_FID_CKPT_DIR:  directory that *contains* a `jax_fid/` sub-directory with
#       the weights file, e.g. JAX_FID_CKPT_DIR=~/qiangchen/data/fid
JAX_FID_CKPT_PATH = "JAX_FID_CKPT_PATH"
JAX_FID_CKPT_DIR = "JAX_FID_CKPT_DIR"


def download(url, ckpt_dir=None):
    name = url[url.rfind("/") + 1 : url.rfind("?")]

    # 1) explicit file path (recommended on clusters without Dropbox access)
    env_path = os.environ.get(JAX_FID_CKPT_PATH, "")
    if env_path and os.path.exists(env_path):
        print(f"Using local FID weights: {env_path}")
        return env_path

    # 2) explicit directory that contains jax_fid/<name>
    ckpt_dir = os.environ.get(JAX_FID_CKPT_DIR, ckpt_dir)
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, "jax_fid")
    ckpt_file = os.path.join(ckpt_dir, name)
    if os.path.exists(ckpt_file):
        print(f"Using local FID weights: {ckpt_file}")
        return ckpt_file

    print(f'Downloading: "{url[:url.rfind("?")]}" to {ckpt_file}')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    delay = random.uniform(0, 20)
    print(f"Sleeping {delay:.2f}s before download...")
    time.sleep(delay)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    # first create temp file, in case the download fails
    ckpt_file_temp = os.path.join(ckpt_dir, name + ".temp")
    with open(ckpt_file_temp, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("An error occured while downloading, please try again.")
        if os.path.exists(ckpt_file_temp):
            os.remove(ckpt_file_temp)
    else:
        # if download was successful, rename the temp file
        os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]
