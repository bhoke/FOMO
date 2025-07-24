import tensorflow as tf
from keras import ops


def weighted_xent(weights):
    """
    Constructs a weighted cross-entropy loss for segmentation tasks.

    Args:
        class_weights: Weight for object class vs background

    Returns:
        Weighted cross-entropy loss function for segmentation
    """
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    def weighted_segmentation_loss(y_true, y_pred):
        # Ensure numerical stability
        y_pred = ops.clip(y_pred, 1e-8, 1.0)
        # Compute log(p)
        log_pred = ops.log(y_pred)
        # Multiply each channel by its class weight
        weights = y_true * weights  # shape: (batch, H, W, num_classes)
        # Cross-entropy: -sum(y_true * log(y_pred) * class_weights) over classes
        loss = -ops.sum(weights * log_pred, axis=-1)
        return ops.mean(loss)

    return weighted_segmentation_loss


def weighted_dice_loss(weights, smooth=1e-5):
    """
    weights: Tensor or array of shape (n_classes,)
    ground truths and predictions should be one-hot encoded.
    """
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        axes = [0, 1, 2]  # sum over batch, height, width
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice_score = (2. * intersection + smooth) / (union + smooth)
        weighted_dice = weights * dice_score
        loss = 1. - tf.reduce_sum(weighted_dice) / tf.reduce_sum(weights)
        return loss
    return loss
