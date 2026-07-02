import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, root_dir, classes=("benign", "malignant"), img_size=256):
        self.img_size = img_size
        self.samples = []  # each entry: (image_path, [mask_path, mask_path_1, ...])

        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            files = os.listdir(cls_dir)
            image_files = [f for f in files if "mask" not in f]

            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                name, _ = os.path.splitext(img_file)
                # some images have multiple masks: name_mask.png, name_mask_1.png
                mask_files = sorted([f for f in files if f.startswith(name + "_mask")])
                mask_paths = [os.path.join(cls_dir, m) for m in mask_files]
                self.samples.append((img_path, mask_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_paths = self.samples[idx]

        # load and resize image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # merge all masks for this image into one binary mask
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for mp in mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.img_size, self.img_size))
            mask = np.maximum(mask, m)

        # normalize
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW, numpy -> tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)  # add channel dim: (1, H, W)

        return image, mask