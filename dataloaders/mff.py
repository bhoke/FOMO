
import os.path as osp
import tensorflow as tf
from .base import Dataset
import json

class MFFDataset(Dataset):
    def __init__(self, cfg, sub_path, augment):
        super().__init__(cfg, sub_path, augment)
        json_path = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME, f"{sub_path}_labels.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.LABEL_DICT = {"fruitfly": 1}
        self.bboxes_list = []

    def load_dataset(self):
        """
        Loads a tf.data.Dataset yielding (image, boxes) pairs.

        Returns:
            tf.data.Dataset
        """
        for file_info in self.data["files"]:
            im_bboxes = []
            filename = file_info["path"]
            self.image_paths.append(osp.join(self.image_dir, filename))
            input2model_ratio = self.cfg.TRAIN.IMAGE_SIZE[0] / self.cfg.DATASET.IMAGE_SIZE[0]
            for bbox in file_info.get("boundingBoxes", []):
                tl_x = int(bbox["x"]) * input2model_ratio // 8
                tl_y = int(bbox["y"]) * input2model_ratio // 8
                br_x = tl_x + int(bbox["width"]) * input2model_ratio // 8
                br_y = tl_y + int(bbox["height"]) * input2model_ratio // 8
                label = int(self.LABEL_DICT[bbox["label"]])
                im_bboxes.append([tl_x, tl_y, br_x, br_y, label])
            self.bboxes_list.append(im_bboxes)

        boxes_dense = tf.ragged.constant(self.bboxes_list).to_tensor(default_value=0)
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, boxes_dense))

        dataset = dataset.map(
            lambda img_path, bboxes: self._bbox2segm(img_path, bboxes),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset
