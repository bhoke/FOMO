import numpy as np
from keras import ops


def weighted_xent(class_weights):
    """
    Constructs a weighted cross-entropy loss for segmentation tasks.

    Args:
        class_weights: Weight for object class vs background

    Returns:
        Weighted cross-entropy loss function for segmentation
    """
    np.array(class_weights)
    def weighted_segmentation_loss(y_true, y_pred):
        # Ensure numerical stability
        y_pred = ops.clip(y_pred, 1e-8, 1.0)
        # Compute log(p)
        log_pred = ops.log(y_pred)
        # Multiply each channel by its class weight
        weights = y_true * class_weights  # shape: (batch, H, W, num_classes)
        # Cross-entropy: -sum(y_true * log(y_pred) * class_weights) over classes
        loss = -ops.sum(weights * log_pred, axis=-1)
        return ops.mean(loss)

    return weighted_segmentation_loss
