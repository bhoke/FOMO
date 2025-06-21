import os.path as osp
from utils import data_utils
import json
import random
import tensorflow as tf
from abc import abstractmethod


class Dataset:
    def __init__(self, cfg, sub_path, augment=False):
        self.cfg = cfg
        self.dataset_dir = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
        if not osp.exists(self.dataset_dir):
            data_utils.download_dataset(cfg.DATASET.ROOT, cfg.DATASET.URL)
        self.image_dir = osp.join(self.dataset_dir, sub_path)
        self.sub_path = sub_path
        self.augment = augment
        self.image_paths = []

    @abstractmethod
    def load_dataset(self):
        pass

    def _bbox2segm(self, image_path, bboxes_list):
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.cfg.TRAIN.IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0

        # Initialize label image (note: 8x smaller than original due to //8 division)
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

    def _poly2segm(image_path, polygons):
        """
        Maps image path and polygons to (image, mask) pair.
        
        Args:
            image_path: String tensor with path to image
            polygons: RaggedTensor with shape [num_polygons, variable_length]
                    Each polygon: [vertices..., label, rle_flag]
        
        Returns:
            (image, mask): Tuple of processed image and label mask
        """
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Get image dimensions
        image_shape = tf.shape(image)
        height, width = image_shape[0], image_shape[1]
        
        # Initialize mask with zeros (background class)
        mask = tf.zeros([height, width], dtype=tf.int32)
        
        # Process each polygon
        def process_polygon(polygon):
            # Extract components from polygon
            polygon_data = tf.cast(polygon, tf.float32)
            rle_flag = tf.cast(polygon_data[-1], tf.int32)
            label = tf.cast(polygon_data[-2], tf.int32)
            vertices_or_rle = polygon_data[:-2]
            
            # Branch based on RLE flag
            return tf.cond(
                tf.equal(rle_flag, 0),
                lambda: create_polygon_mask(vertices_or_rle, label, height, width),
                lambda: decode_rle_mask(vertices_or_rle, label, height, width)
            )
        
        # Apply processing to each polygon and combine masks
        polygon_masks = tf.map_fn(
            process_polygon,
            polygons,
            fn_output_signature=tf.TensorSpec([None, None], dtype=tf.int32),
            parallel_iterations=10
        )
        
        # Combine all polygon masks (later labels override earlier ones)
        def combine_masks(current_mask, new_mask):
            # Only update pixels where new_mask is non-zero
            return tf.where(tf.equal(new_mask, 0), current_mask, new_mask)
        
        final_mask = tf.foldl(combine_masks, polygon_masks, initializer=mask)
        
        return image, final_mask

    def create_polygon_mask(vertices, label, height, width):
        """
        Creates a mask from polygon vertices using point-in-polygon algorithm.
        """
        # Reshape vertices to [num_points, 2] format
        num_vertices = tf.shape(vertices)[0] // 2
        points = tf.reshape(vertices, [num_vertices, 2])
        
        # Create coordinate grids
        y_coords = tf.range(height, dtype=tf.float32)
        x_coords = tf.range(width, dtype=tf.float32)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten coordinates for point-in-polygon test
        pixel_coords = tf.stack([tf.reshape(yy, [-1]), tf.reshape(xx, [-1])], axis=1)
        
        # Point-in-polygon test using ray casting algorithm
        def point_in_polygon(point):
            x, y = point[1], point[0]  # Note: swap for x,y convention
            
            # Ray casting algorithm
            inside = tf.constant(False)
            j = num_vertices - 1
            
            def body(i, j, inside):
                xi, yi = points[i][0], points[i][1]
                xj, yj = points[j][0], points[j][1]
                
                condition = tf.logical_and(
                    tf.logical_or(
                        tf.logical_and(yi > y, yj <= y),
                        tf.logical_and(yj > y, yi <= y)
                    ),
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi
                )
                
                inside = tf.logical_xor(inside, condition)
                return i + 1, i, inside
            
            _, _, final_inside = tf.while_loop(
                lambda i, j, inside: i < num_vertices,
                body,
                [0, j, inside]
            )
            
            return final_inside
        
        # Apply point-in-polygon test to all pixels
        inside_mask = tf.map_fn(
            point_in_polygon,
            pixel_coords,
            fn_output_signature=tf.TensorSpec([], dtype=tf.bool),
            parallel_iterations=1000
        )
        
        # Reshape back to image dimensions and apply label
        inside_mask = tf.reshape(inside_mask, [height, width])
        polygon_mask = tf.where(inside_mask, label, 0)
        
        return tf.cast(polygon_mask, tf.int32)

    def decode_rle_mask(rle_data, label, height, width):
        """
        Decodes RLE (Run Length Encoding) data to create a mask.
        """
        # Convert RLE data to integers
        rle_ints = tf.cast(rle_data, tf.int32)
        
        # Split into starts and lengths (assuming alternating format)
        starts = rle_ints[::2] - 1  # Convert to 0-based indexing
        lengths = rle_ints[1::2]
        
        # Calculate total size
        total_size = height * width
        
        # Create flat mask
        flat_mask = tf.zeros([total_size], dtype=tf.int32)
        
        # Fill in the RLE segments
        def fill_segment(i, mask):
            start = starts[i]
            length = lengths[i]
            end = start + length
            
            # Create indices for this segment
            indices = tf.range(start, end)
            
            # Create updates (label values)
            updates = tf.fill([length], label)
            
            # Update mask
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.expand_dims(indices, 1),
                updates
            )
            
            return i + 1, mask
        
        # Apply all RLE segments
        num_segments = tf.shape(starts)[0]
        _, final_mask = tf.while_loop(
            lambda i, mask: i < num_segments,
            fill_segment,
            [0, flat_mask]
        )
        
        # Reshape to image dimensions
        return tf.reshape(final_mask, [height, width])


if __name__ == "__main__":
    dataset_path = osp.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    train_ds = load_dataset(dataset_path, "train")
    for img, (boxes, labels) in train_ds.take(1):
        print(img.shape, boxes, labels)
