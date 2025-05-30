
import os.path as osp
import tensorflow as tf
from .base import Dataset
import json

class MMFDataset(Dataset):
    def __init__(self, cfg, dataset_type, augment):
        super().__init__(cfg, dataset_type, augment)
        self.cfg = cfg
        self.image_dir = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, dataset_type)
        json_path = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, f"{dataset_type}_labels.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.LABEL_DICT = {"fruitfly": 1}

    def load_dataset(self):
        """
        Loads a tf.data.Dataset yielding (image, boxes) pairs.

        Args:
            root_dir (str): Path to dataset root directory.
            dataset_type (str): 'train' or 'test'.
            img_height (int): Desired image height after resizing.
            img_width (int): Desired image width after resizing.

        Returns:
            tf.data.Dataset
        """
        for file_info in self.data["files"]:
            im_bboxes = []
            filename = file_info["path"]
            self.image_paths.append(osp.join(self.image_dir, filename))
            input2model_ratio = self.cfg.TRAIN.IMAGE_SIZE[0] / self.cfg.DATASET.IMAGE_SIZE[0]
            for bbox in file_info.get("boundingBoxes", []):
                tl_x, tl_y, br_x, br_y, label = (
                    max(int(bbox["x"]) * input2model_ratio // 8, 0),
                    max(int(bbox["y"]) * input2model_ratio // 8, 0),
                    min(
                        int(bbox["x"]) * input2model_ratio // 8
                        + int(bbox["width"]) // 8,
                        self.cfg.TRAIN.IMAGE_SIZE[0] // 8 - 1,
                    ),
                    min(
                        int(bbox["y"]) * input2model_ratio // 8
                        + int(bbox["height"]) // 8,
                        self.cfg.TRAIN.IMAGE_SIZE[1] // 8 - 1,
                    ),
                    int(self.LABEL_DICT[bbox["label"]]),
                )
                im_bboxes.append([tl_x, tl_y, br_x, br_y, label])
            self.bboxes_list.append(im_bboxes)

        boxes_dense = tf.ragged.constant(self.bboxes_list).to_tensor(default_value=0)
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, boxes_dense))

        dataset = dataset.map(
            lambda img_path, bboxes: self._process_example(img_path, bboxes, self.augment),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset
