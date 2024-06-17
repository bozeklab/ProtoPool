import os
import random
import pathlib
import glob
import numpy as np
import shutil


def get_dir_list(path):
    dirs = glob.glob(path, recursive=True)
    dirs.sort()
    return dirs


path = 'mito_scale_resized_512/**/*.png'
dirs = np.array(get_dir_list(path))
print(dirs.shape)
print(dirs)

val_test_size = (0., 0.3)
random_state = 3

length = len(dirs)
result_list = np.zeros(length, dtype=int)
splits = [int(length * size) for size in val_test_size]
result_list[:splits[0]] = 1
result_list[splits[0]:splits[0] + splits[1]] = 2
np.random.seed(random_state)
np.random.shuffle(result_list)

for (i, mode) in enumerate(['train', 'val', 'test']):
    to_save = dirs[result_list == i]
    for file in to_save:
        out_file = file.replace('mito_scale_resized_512', 'mito_scale_resized_512_split/' + mode)
        print(out_file)
        os.makedirs(pathlib.Path(out_file).parent, exist_ok=True)
        shutil.copy(file, out_file)

