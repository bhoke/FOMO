from gc import callbacks
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.applications import MobileNetV2
from keras.layers import BatchNormalization, Conv2D, Softmax
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from utils import bbox2segm
from losses import DiceLoss


BEST_MODEL_PATH = "fomo.h5"
CLASSES = 1
MODEL_INPUT_SHAPE = (320, 320, 3)
BATCH_SIZE = 16
tf.random.set_seed(42)

def build_model(input_shape: tuple, weights: str, alpha: float,
                num_classes: int) -> tf.keras.Model:
    """ Construct a constrained object detection model.

    Args:
        input_shape: Passed to MobileNet construction.
        weights: Weights for initialization of MobileNet where None implies
            random initialization.
        alpha: MobileNet alpha value.
        num_classes: Number of classes, i.e. final dimension size, in output.

    Returns:
        Uncompiled keras model.

    Model takes (B, H, W, C) input and
    returns (B, H//8, W//8, num_classes) logits.
    """

    #! First create full mobile_net_V2 from (HW, HW, C) input
    #! to (HW/8, HW/8, C) output
    mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                weights=weights,
                                alpha=alpha,
                                include_top=True)
    #! Default batch norm is configured for huge networks, let's speed it up
    for layer in mobile_net_v2.layers:
        if type(layer) == BatchNormalization:
            layer.momentum = 0.9
    #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
    cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
    #! Now attach a small additional head on the MobileNet
    model = Conv2D(filters=32, kernel_size=1, strides=1,
                activation='relu', name='head')(cut_point.output)
    logits = Conv2D(filters=num_classes, kernel_size=1, strides=1,
                    activation=None, name='logits')(model)
    return Model(inputs=mobile_net_v2.input, outputs=logits)
def train(num_classes: int, learning_rate: float, num_epochs: int,
          alpha: float,
          train_images: np.ndarray,
          train_labels: np.ndarray,
          validation_split: float,
          best_model_path: str,
          input_shape: tuple) -> tf.keras.Model:
    """ Construct and train a constrained object detection model.

    Args:
        num_classes: Number of classes in datasets. This does not include
            implied background class introduced by segmentation map dataset
            conversion.
        learning_rate: Learning rate for Adam.
        num_epochs: Number of epochs passed to model.fit
        alpha: Alpha used to construct MobileNet. Pretrained weights will be
            used if there is a matching set.
        object_weight: The weighting to give the object in the loss function
            where background has an implied weight of 1.0.
        train_dataset: Training dataset of (x, (bbox, one_hot_y))
        validation_dataset: Validation dataset of (x, (bbox, one_hot_y))
        best_model_path: location to save best model path. note: weights
            will be restored from this path based on best val_f1 score.
        input_shape: The shape of the model's input
        max_training_time_s: Max training time (will exit if est. training time is over the limit)
        is_enterprise_project: Determines what message we print if training time exceeds
    Returns:
        Trained keras model.

    Constructs a new constrained object detection model with num_classes+1
    outputs (denoting the classes with an implied background class of 0).
    Both training and validation datasets are adapted from
    (x, (bbox, one_hot_y)) to (x, segmentation_map). Model is trained with a
    custom weighted cross entropy function.
    """
    num_classes_with_background = num_classes + 1

    width, height, _ = input_shape
    if width != height:
        raise Exception(f"Only square inputs are supported; not {input_shape}")

    model = build_model(
        input_shape=input_shape,
        weights=None,
        alpha=alpha,
        num_classes=num_classes_with_background
    )

    #! Derive output size from model
    model_output_shape = model.layers[-1].output.shape
    _batch, width, height, num_classes = model_output_shape
    if width != height:
        raise Exception(f"Only square outputs are supported; not {model_output_shape}")

    callbacks = [ModelCheckpoint(best_model_path, save_best_only=True, verbose = 1)]
    model.compile(loss=DiceLoss,
                  optimizer=Adam(learning_rate=learning_rate))

    model.fit(train_images, train_labels,
              validation_split = validation_split,
              batch_size = BATCH_SIZE,
              callbacks = callbacks,
              epochs=num_epochs, verbose=1)

    #! Restore best weights.
    model.load_weights(best_model_path)

    #! Add explicit softmax layer before export.
    softmax_layer = Softmax()(model.layers[-1].output)
    model = Model(model.input, softmax_layer)

    return model

if __name__ == '__main__':

    ams_data = np.load('data/ams_data.npy')
    num_samples = ams_data.shape[0]
    ams_data = tf.convert_to_tensor(ams_data.reshape((num_samples, *MODEL_INPUT_SHAPE)))[:-5]
    segm_labels = tf.convert_to_tensor(bbox2segm(ams_data, MODEL_INPUT_SHAPE, (40,40), 2))
        
    model = train(num_classes=CLASSES,
                learning_rate=1e-4,
                num_epochs=60,
                alpha=0.1,
                train_images = ams_data,
                train_labels = segm_labels,
                validation_split = 0.1,
                best_model_path=BEST_MODEL_PATH,
                input_shape=MODEL_INPUT_SHAPE)
