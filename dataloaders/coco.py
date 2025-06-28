import math
import json
import os.path as osp
from collections import defaultdict

import cv2
import keras
import numpy as np


class COCODataset(keras.utils.PyDataset):
    def __init__(self, cfg, sub_path, augment, **kwargs):
        super().__init__(**kwargs)
        self.img_size = cfg.DATASET.IMAGE_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.image_dir = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path)
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.mask_size = tuple(size // 8 for size in self.img_size)
        json_path = osp.join(
            cfg.DATASET.ROOT,
            cfg.DATASET.NAME,
            "annotations",
            f"instances_{sub_path}.json",
        )
        with open(json_path, "r") as f:
            json_data = json.load(f)
        imgToAnns = {}
        image_id_to_info = {img["id"]: img for img in json_data["images"]}
        for ann in json_data["annotations"]:
            if not imgToAnns.get(ann["image_id"]):
                file_info = image_id_to_info[ann["image_id"]]
                imgToAnns[ann["image_id"]] = {
                    "file_specs": {
                        "file_name": file_info["file_name"],
                        "height": file_info["height"],
                        "width": file_info["width"],
                    },
                    "annotations": [],
                }
            voi = {key: ann[key] for key in ("segmentation", "iscrowd", "category_id")}
            imgToAnns[ann["image_id"]]["annotations"].append(voi)
        self.category_id_to_label = {
            cat["id"]: i for i, cat in enumerate(json_data["categories"])
        }
        self.img_ids = list(imgToAnns.keys())
        self.data = list(imgToAnns.values())

    def __len__(self):
        return math.ceil(len(self.img_ids) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.img_ids))
        batch_img_ids = self.img_ids[low:high]
        batch_data = self.data[low:high]
        data_size = len(batch_data)
        imgs = np.zeros(((data_size,) + self.img_size + (3,)), np.float32)
        masks = np.zeros(((data_size,) + self.mask_size + (self.num_classes,)), np.uint8)

        for datum_idx, datum in enumerate(batch_data):
            img_path = osp.join(self.image_dir, datum["file_specs"]["file_name"])
            img = cv2.imread(img_path)[..., ::-1].astype(np.float32) / 255.0
            img = cv2.resize(img, self.img_size)
            width = datum["file_specs"]["width"]
            height = datum["file_specs"]["height"]
            mask = np.zeros((self.mask_size + (self.num_classes,)), np.uint8)
            for ann in datum["annotations"]:
                label = self.category_id_to_label[ann["category_id"]]
                segm = ann["segmentation"]
                if ann["iscrowd"]:
                    rle_mask = self._rle_decode(segm)
                    mask[..., label] += rle_mask
                else:
                    pts = [None] * len(segm)
                    for seg_idx in range(len(pts)):
                        poly_points = np.reshape(segm[seg_idx], (-1, 2))
                        poly_points[:, 0] *= self.mask_size[1] / width
                        poly_points[:, 1] *= self.mask_size[1] / height
                        pts[seg_idx] = poly_points.astype(np.int32)
                    poly_mask = cv2.fillPoly(mask[..., label].copy(), pts, 1)
                    mask[..., label] = cv2.resize(poly_mask, self.mask_size)
            mask = cv2.resize(mask, self.mask_size)
            imgs[datum_idx] = img
            masks[datum_idx] = mask

        return imgs, masks

    def _rle_decode(self, rle_data):
        rle_ints = np.array(rle_data["counts"], dtype=np.int32)
        width, height = rle_data["size"][1], rle_data["size"][0]
        total_size = width * height

        # Split into starts and lengths (assuming alternating format)
        starts = np.cumsum(rle_ints)[:-1:2]
        lengths = rle_ints[1::2]
        flat_mask = np.zeros([total_size], dtype=np.uint8)

        for i in range(len(starts)):
            start = starts[i]
            length = lengths[i]
            end = start + length
            indices = np.arange(start, end)
            flat_mask[indices] = 1

        mask = flat_mask.reshape(height, width)
        mask = cv2.resize(mask, self.mask_size)
        return mask
        
