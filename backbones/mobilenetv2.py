from keras import Model
from keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    DepthwiseConv2D,
    Add,
    ZeroPadding2D,
    Input,
)
from keras.utils import get_file

BASE_WEIGHT_PATH = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _correct_pad(input_shape, kernel_size: int):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

    Returns:
    A tuple.
    """
    input_size = input_shape[2:4]
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size // 2, kernel_size // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def _inverted_res(inputs, expansion, stride, alpha, filters, block_id):
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    prefix = f"block_{block_id}_"
    input_shape = inputs.shape
    num_channels = input_shape[3]
    x = inputs

    if block_id:
        # Expand with a pointwise 1x1 convolution.
        x = Conv2D(
            expansion * 3,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )(inputs)
        x = BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )(x)
        x = ReLU(6.0, name=prefix + "expand_relu")(x)
        if block_id == 6:
            return x
    else:
        prefix = "expanded_conv_"

    # Depthwise 3x3 convolution.
    if stride == 2:
        x = ZeroPadding2D(padding=_correct_pad(input_shape, 3), name=prefix + "pad")(x)
    x = DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )(x)
    x = BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )(x)

    x = ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project with a pointwise 1x1 convolution.
    x = Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x)
    x = BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )(x)

    if pointwise_filters == num_channels and stride == 1:
        x = Add(name=prefix + "add")((x, inputs))
    return x


def MobileFOMOv2(input_shape, alpha, num_classes, weights=None):
    input_shape = input_shape + (3,) if len(input_shape) == 2 else input_shape
    first_block_filters = _make_divisible(32 * alpha, 8)
    img_input = Input(shape=input_shape)
    x = Conv2D(
        first_block_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
    )(img_input)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name="bn_Conv1")(x)
    x = ReLU(6, name="Conv1_relu")(x)

    x = _inverted_res(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    x = _inverted_res(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)
    x = _inverted_res(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)
    x = _inverted_res(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu', name='head')(x)
    logits = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, name='logits')(x)
    model = Model(inputs=img_input, outputs=logits)

    if weights == "imagenet":
        model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(float(alpha))
                + "_"
                + str(input_shape[1])
                + "_no_top"
                + ".h5"
            )
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(
            model_name, weight_path, cache_subdir="models"
        )
        model.load_weights(weights_path, by_name=True)

    return model

if __name__ == "__main__":
    inp_shape = (224, 224, 3)
    model = MobileFOMOv2(inp_shape, 1.0, 5, "imagenet")
    model.compile()
    model.summary()
