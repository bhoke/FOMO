import keras
from keras import Model
from keras.layers import Conv2D

def MobileFOMOv3(input_shape, alpha, num_classes, weights="imagenet"):
    """
    FOMO implementation with MobileNetV3 Small backbone.
    """
    input_shape = tuple(input_shape)
    input_shape = input_shape + (3,) if len(input_shape) == 2 else input_shape
    # MobileNetV3Small imagenet weights only support alpha 0.75 and 1.0 in some Keras versions
    # We'll check if alpha is supported for imagenet weights
    backbone_weights = weights if (weights == "imagenet" and alpha in [0.75, 1.0]) else None
    if weights == "imagenet" and backbone_weights is None:
        print(f"Warning: MobileNetV3 imagenet weights are only supported for alpha=0.75 or 1.0. Using alpha={alpha} without pretrained weights.")

    base_model = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=backbone_weights
    )

    # For MobileNetV3Small, expanded_conv_3_expand is the 1/8 resolution layer
    try:
        x = base_model.get_layer("expanded_conv_3_expand").output
    except ValueError:
        # Fallback dynamic selection of 1/8 resolution layer
        target_size = (input_shape[0] // 8, input_shape[1] // 8)
        selected_layer = None
        for layer in base_model.layers:
            if layer.output.shape[1:3] == target_size:
                selected_layer = layer
            elif selected_layer is not None and layer.output.shape[1:3] != target_size:
                break
        x = selected_layer.output

    # FOMO head
    x = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu', name='head')(x)
    logits = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation="softmax", name='out')(x)

    model = Model(inputs=base_model.input, outputs=logits)

    if weights is not None and weights != "imagenet":
        model.load_weights(weights)
        print(f"Previous model restored from {weights}")

    return model

if __name__ == "__main__":
    model = MobileFOMOv3((224, 224, 3), 1.0, 2)
    model.summary()
