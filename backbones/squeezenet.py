from keras.layers import Input, Conv2D, MaxPooling2D, Rescaling, concatenate
from keras import Model


def fire_module(prev_layer, layer_name, squeeze, expand):
    squeeze_layer = Conv2D(
        filters=squeeze,
        kernel_size=(1, 1),
        strides=1,
        activation="relu",
        padding="valid",
        name=f"{layer_name}_squeeze",
    )(prev_layer)
    expand_1x1 = Conv2D(
        filters=expand,
        kernel_size=(1, 1),
        strides=1,
        activation="relu",
        padding="valid",
        name=f"{layer_name}_exp1x1",
    )(squeeze_layer)
    expand_3x3 = Conv2D(
        filters=expand,
        kernel_size=(3, 3),
        strides=1,
        activation="relu",
        padding="same",
        name=f"{layer_name}_exp3x3",
    )(squeeze_layer)
    return concatenate((expand_1x1, expand_3x3))


def SqueezeFOMO(input_shape, num_classes):
    final_act = "sigmoid" if num_classes == 1 else "softmax"
    model_input = Input(shape=input_shape)
    rescaler = Rescaling(scale=1.0 / 255.0)(model_input)
    conv1 = Conv2D(
        filters=96, kernel_size=(7, 7), strides=2, padding="same", name="conv1"
    )(rescaler)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=2, name="maxpool1", padding="same"
    )(conv1)
    fire2 = fire_module(maxpool1, "fire2", squeeze=16, expand=64)
    fire3 = fire_module(fire2, "fire3", squeeze=16, expand=64)
    fire4 = fire_module(fire3, "fire4", squeeze=32, expand=128)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(fire4)
    preds = Conv2D(
        filters=num_classes, kernel_size=1, activation=None, strides=1, name="preds"
    )(maxpool4)
    return Model(model_input, preds)
