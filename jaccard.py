import os
import glob
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
import torch
import torchvision.transforms.functional as F
import numpy as np

def create_boolean_mask(mask_img):
    mask_img = F.rgb_to_grayscale(mask_img)
    m_shape = mask_img.shape
    if len(m_shape) == 3:
        mask_img=mask_img.squeeze()
    else:
        mask_img=mask_img[:, 0, :, :]
    mask = mask_img.numpy()
    if mask.max() - mask.min() < 0.0001:
        return torch.zeros(mask_img.shape).bool()

    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).astype(np.uint8)
    #max_val = np.max(mask)
    #min_val = np.min(mask)
    #max_mask = (mask == max_val)
    #min_mask = (mask == min_val)
    bool_mask = mask > 100
    return torch.tensor(bool_mask)


# Define the root directory where you want to search
root_dir = "/data/pwojcik/ProtoPool/results/img_proto/mito_descriptive-10_prototypes-50_lr-0.001_resnet18_True_log_log_warmup_ll_seed-0_2024-10-03_224955/"

# Use glob to search for files matching the pattern in all subdirectories
pattern = os.path.join(root_dir, '**', '*_mask_*.png')

# Use glob with recursive search
files = glob.glob(pattern, recursive=True)

# Filter out files that match the "nearest-6_" pattern
filtered_files = [file_path for file_path in files if not os.path.basename(file_path).startswith('nearest-6_')]

jaccard = []

# Print the filtered list of files
for file_path in filtered_files:
    print(file_path)
    mask = Image.open(file_path).convert("RGB")
    msk_tensor = transforms.ToTensor()(mask)
    bool_mask = create_boolean_mask(msk_tensor)
    num_white_pixels = torch.sum(bool_mask).item()
    jaccard.append(num_white_pixels/(bool_mask.shape[0] * bool_mask.shape[1]))

import statistics
print('Jaccard: ', statistics.mean(jaccard))