import tensorflow as tf
from pathlib import Path

class VIRATDataset:
    def __init__(self, cfg, sub_path, augment, shuffle=False, **kwargs):
        self.cfg = cfg
        self.sub_path = sub_path
        self.augment = augment
        self.shuffle = shuffle
        self.image_dir = Path(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path, "images")
        self.ann_dir = Path(cfg.DATASET.ROOT, cfg.DATASET.NAME, sub_path, "annotations")
        self.images = sorted([str(f) for f in self.image_dir.iterdir() if f.is_file()])
        self.annotations = sorted([str(f) for f in self.ann_dir.iterdir() if f.is_file()])
        self.img_size = cfg.TRAIN.IMAGE_SIZE
        self.mask_size = (self.img_size[0] // 8, self.img_size[1] // 8)
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def _decode_and_process(self, image_path, mask_path):
        img_size = self.img_size
        mask_size = self.mask_size
        num_classes = self.num_classes

        # Read and decode image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, img_size, method='bilinear')

        # Read and decode mask (grayscale)
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, mask_size, method='nearest')
        mask = tf.squeeze(mask, axis=-1)
        mask = tf.cast(mask, tf.int32)
        mask = tf.one_hot(mask, num_classes)

        if self.augment:
            # Random brightness and contrast on image
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

            # Synchronized random horizontal flip
            flip = tf.random.uniform(()) > 0.5
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
        return img, mask

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.annotations))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images), reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda x, y: self._decode_and_process(x, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
