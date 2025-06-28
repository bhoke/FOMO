import os.path as osp
import math
import json

import numpy as np
import keras
import cv2

class MFFDataset(keras.utils.PyDataset):
    def __init__(self, cfg, sub_path, augment, **kwargs):
        super().__init__(**kwargs)
        json_path = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, f"{sub_path}_labels.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)["files"]
        self.LABEL_DICT = {"fruitfly": 1}
        self.img_size = cfg.DATASET.IMAGE_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.image_dir = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path)
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.mask_size = tuple(size // 8 for size in self.img_size)
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.data))
        batch_img_ids = self.data[low:high]
        batch_data = self.data[low:high]
        data_size = len(batch_data)
        imgs = np.zeros(((data_size,) + self.img_size + (3,)), np.float32)
        masks = np.zeros(((data_size,) + self.mask_size + (self.num_classes,)), np.uint8)
        for file_idx, file_info in enumerate(batch_data):
            filename = file_info["path"]
            img_path = osp.join(self.image_dir, filename)
            img = cv2.imread(img_path)[..., ::-1].astype(np.float32) / 255.0
            img = cv2.resize(img, self.img_size)
            height, width = img.shape[:2]
            mask = np.zeros(self.mask_size + (self.num_classes,), np.uint8)
            for bbox in file_info.get("boundingBoxes", []):
                tl_x = int(bbox["x"] * 8 / width)
                tl_y = int(bbox["y"] * 8 / height)
                br_x = tl_x + int(bbox["width"] * 8 / width)
                br_y = tl_y + int(bbox["height"] * 8 / height)
                mask = cv2.rectangle(mask, (tl_x, tl_y), (br_x, br_y), 1, cv2.FILLED)
            imgs[file_idx] = img
            masks[file_idx] = mask
        return imgs, masks
        
        # if self.augment:
        #     seed = (random.randint(0, 10), random.randint(0, 10))
        #     image = tf.image.stateless_random_flip_left_right(image, seed)
        #     image = tf.image.stateless_random_flip_up_down(image, seed)
        #     image = tf.image.random_brightness(image, 0.2)
        #     image = tf.image.random_contrast(image, 0.8, 1.2)
        #     image = tf.image.random_saturation(image, 0.8, 1.2)

        #     label_image = tf.image.stateless_random_flip_left_right(label_image, seed)
        #     label_image = tf.image.stateless_random_flip_up_down(label_image, seed)

        # return image, label_image
