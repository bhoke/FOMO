import argparse
import importlib

import tensorflow as tf
from keras import Model
from keras import optimizers
from keras.layers import Softmax
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint

import backbones
from configs import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train FOMO network")

    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        default="configs/mff/mff_mobilenetv2.yaml",
        type=str,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args


def main() -> Model:
    args = parse_args()
    dataloader_module = importlib.import_module("dataloaders")
    DatasetClass = getattr(dataloader_module, config.DATASET.NAME)
    
    train_ds = DatasetClass(config, "train", augment = True)
    val_ds = DatasetClass(config, "test", augment = False)

    train_dataloader = (
        train_ds.load_dataset()
        .shuffle(config.TRAIN.BATCH_SIZE * 10)
        .batch(config.TRAIN.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataloader = (
        val_ds.load_dataset()
        .batch(config.TRAIN.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    if config.MODEL.BACKBONE.lower() == "mobilenetv2":
        model = backbones.MobileFOMOv2(
            config.TRAIN.IMAGE_SIZE, 0.35, config.DATASET.NUM_CLASSES, "imagenet"
        )
    elif config.MODEL.BACKBONE.lower() == "squeezenet":
        model = backbones.SqueezeFOMO(
            config.TRAIN.IMAGE_SIZE, config.DATASET.NUM_CLASSES
        )
    else:
        print("Invalid model name or model not implemented yet!")
        return None

    #! Derive output size from model
    model_output_shape = model.layers[-1].output.shape
    _, width, height, _ = model_output_shape
    if width != height:
        raise Exception(f"Only square outputs are supported; not {model_output_shape}")

    optim: optimizers.Optimizer = optimizers.get(config.TRAIN.OPTIMIZER)
    optim.learning_rate = config.TRAIN.LR
    callbacks = [
        ModelCheckpoint(config.TRAIN.BEST_SAVE_PATH, save_best_only=True, verbose=1)
    ]
    model.compile(loss=MeanSquaredError(), optimizer=optim)

    model.fit(
        train_dataloader,
        batch_size=config.TRAIN.BATCH_SIZE,
        callbacks=callbacks,
        epochs=config.TRAIN.NUM_EPOCHS,
        verbose=1,
        validation_data=val_dataloader,
    )

    #! Restore best weights.
    model.load_weights(config.TRAIN.BEST_SAVE_PATH)

    #! Add explicit softmax layer before export.
    softmax_layer = Softmax()(model.layers[-1].output)
    model = Model(model.input, softmax_layer)

    return model


if __name__ == "__main__":
    main()
