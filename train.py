import argparse
import importlib

import tensorflow as tf
from keras import Model
from keras import optimizers
from utils.losses import weighted_xent
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


def get_model_by_name(model_name):
    if model_name == "mobilenetv2":
        model = backbones.MobileFOMOv2(
            config.TRAIN.IMAGE_SIZE, 0.35, config.DATASET.NUM_CLASSES, "imagenet"
        )
    elif model_name == "squeezenet":
        model = backbones.SqueezeFOMO(
            config.TRAIN.IMAGE_SIZE, config.DATASET.NUM_CLASSES
        )
    # elif model_name == "mobilenetv3":
    #     print("Model not implemented yet")
    # elif model_name == "mobilevit":
    #     print("Model not implemented yet")
    else:
        print("Invalid model name or model not implemented yet!")
        raise NotImplementedError
    return model


def main() -> Model:
    args = parse_args()
    dataloader_module = importlib.import_module("dataloaders")
    DatasetClass = getattr(dataloader_module, config.DATASET.NAME)

    train_ds = DatasetClass(
        config,
        config.DATASET.TRAIN_SET,
        augment=True,
        shuffle=True,
        workers=4,
        use_multiprocessing=True,
    )
    val_ds = DatasetClass(
        config,
        config.DATASET.VALIDATION_SET,
        augment=False,
        workers=4,
        use_multiprocessing=True,
    )

    if hasattr(train_ds, "get_dataset"):
        train_ds = train_ds.get_dataset()
        val_ds = val_ds.get_dataset()

    model = get_model_by_name(config.MODEL.BACKBONE.lower())

    optim: optimizers.Optimizer = optimizers.get(config.TRAIN.OPTIMIZER)
    optim.learning_rate = config.TRAIN.LR
    callbacks = [
        ModelCheckpoint(config.TRAIN.BEST_SAVE_PATH, save_best_only=True, verbose=1)
    ]
    model.compile(loss=weighted_xent(config.TRAIN.CLASS_WEIGHTS), optimizer=optim)

    model.fit(
        train_ds,
        batch_size=config.TRAIN.BATCH_SIZE,
        callbacks=callbacks,
        epochs=config.TRAIN.NUM_EPOCHS,
        verbose=1,
        validation_data=val_ds,
    )


if __name__ == "__main__":
    main()
