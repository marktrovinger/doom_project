import numpy as np
import glob
from pathlib import Path
import os
import gzip
import shutil

'''Some utils to massage data into the format expected by the Decision Transformer (CircularReplayBuffer)
1. Need to strip out the .npy extension
2. Need to gz files
'''

def read_folder(path, search_str):
    files = [f for f in glob.glob(path + search_str, recursive=False)]
    return files

def strip_file_extension(folder):
    # SB3 replay files have .npy extension. Remove this to align with what DT expects
    # Note this will strip out another extension if you re-run on same data...
    files = read_folder(folder , '*.npy')
    new_files = []
    for file in files:
        name = Path(file).stem
        os.rename(file, f"{folder}{name}")
        new_files.append(f"{folder}{name}")
    return new_files

def gzip_files(folder):
    files = read_folder(folder , '*')

    for file in files:
        with open(file, 'rb') as f_in:
            with gzip.open(f"{file}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

folder = '/Users/perusha/tensorboard/DT_dataset/atari_7Feb/'
folder = '/Users/perusha/tensorboard/DT_dataset/atari_9Feb/'
strip_file_extension(folder)
gzip_files(folder)