import os
import json
import random
import tensorflow as tf
from abc import abstractmethod

class Dataset:
    def __init__(self, cfg, dataset_type, augment=False):
        self.cfg = cfg
        self.dataset_type =dataset_type
        self.augment = augment
        self.image_paths = []
        self.bboxes_list = []

    @abstractmethod
    def load_dataset(self):
        pass

    def _process_example(self,image_path, bboxes_list, augment=False):
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.cfg.TRAIN.IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0

        # Initialize label image (note: 8x smaller than original due to your //8 division)
        label_shape = (
            self.cfg.TRAIN.IMAGE_SIZE[0] // 8,
            self.cfg.TRAIN.IMAGE_SIZE[1] // 8,
            1,
        )
        label_image = tf.zeros(label_shape, dtype=tf.uint8)

        # Only use non-padded boxes
        valid = tf.reduce_any(tf.not_equal(bboxes_list, 0), axis=-1)
        bboxes_list = tf.boolean_mask(bboxes_list, valid)

        # Process all boxes using vectorized_map
        label_image = tf.foldl(
            self._update_label_for_bbox,
            bboxes_list,
            initializer=label_image,
        )

        # Data augmentation
        if self.augment:
            seed = (random.randint(0, 10), random.randint(0, 10))
            image = tf.image.stateless_random_flip_left_right(image, seed)
            image = tf.image.stateless_random_flip_up_down(image, seed)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)

            label_image = tf.image.stateless_random_flip_left_right(label_image, seed)
            label_image = tf.image.stateless_random_flip_up_down(label_image, seed)

        return image, label_image

    def _update_label_for_bbox(self, label_image, bbox):
        """Vectorized update for a single bounding box"""
        tl_x, tl_y, br_x, br_y, label = tf.unstack(bbox)
        label = tf.cast(label, tf.uint8)

        # Generate grid coordinates using meshgrid
        rows = tf.range(tl_y, br_y, dtype=tf.int32)
        cols = tf.range(tl_x, br_x, dtype=tf.int32)
        jj, ii = tf.meshgrid(cols, rows)

        # Create indices tensor [N, 3]
        indices = tf.stack([ii, jj, tf.zeros_like(ii)], axis=-1)
        indices = tf.reshape(indices, [-1, 3])

        # Create updates tensor
        updates = tf.ones([tf.shape(indices)[0]], dtype=tf.uint8) * label

        return tf.tensor_scatter_nd_max(label_image, indices, updates)


if __name__ == "__main__":
    dataset_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    train_ds = load_dataset(dataset_path, "train")
    for img, (boxes, labels) in train_ds.take(1):
        print(img.shape, boxes, labels)
