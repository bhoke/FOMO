from pathlib import Path
import math
import json
import random

import numpy as np
import keras
import cv2

from utils.data_utils import Augment


class VIRATDataset(keras.utils.PyDataset):
    def __init__(self, cfg, sub_path, augment, shuffle=False, **kwargs):
        super().__init__(**kwargs)
        image_dir = Path(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path, "images")
        ann_dir = Path(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path, "annotations")
        self.LABEL_DICT = {"person": 1, "car": 2}
        self.img_size = cfg.DATASET.IMAGE_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.mask_size = tuple(size // 8 for size in self.img_size)
        images = sorted(image_dir.iterdir())
        annotations = sorted(ann_dir.iterdir())
        self.data = list(zip(images, annotations))
        self.augment_fn = None
        if shuffle:
            random.shuffle(self.data)
        if augment:
            self.augment_fn = Augment(42)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.data))
        batch_data = self.data[low:high]
        data_size = len(batch_data)
        imgs = np.zeros(((data_size,) + self.img_size + (3,)), np.float32)
        masks = np.zeros(
            ((data_size,) + self.mask_size + (self.num_classes,)), np.uint8
        )
        for file_idx, (image_path, ann_path) in enumerate(batch_data):
            img = cv2.imread(image_path)[..., ::-1].astype(np.float32) / 255.0
            ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, self.img_size[::-1])
            height, width = img.shape[:2]
            mask = cv2.resize(
                ann, self.mask_size[::-1], interpolation=cv2.INTER_NEAREST
            )

            one_hot = np.eye(self.num_classes)[mask]
            imgs[file_idx] = img
            masks[file_idx] = one_hot
        if self.augment_fn:
            imgs, masks = self.augment_fn((imgs, masks))
        return imgs, masks
