# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PIN_MEMORY = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'MobilenetV2'
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_OUTPUTS = 2


_C.LOSS = CN()
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
_C.LOSS.SB_WEIGHTS = 0.5

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'dataset/'
_C.DATASET.NAME = 'mff'
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'test'
_C.DATASET.URL = 'https://www.kaggle.com/api/v1/datasets/download/berkanhoke/mediterranean-fruit-fly-images-dataset'
_C.DATASET.IMAGE_SIZE = (400, 400)  # width * height

# training
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = (224, 224)  # width * height
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.LR = 0.01
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.RESUME = False
_C.TRAIN.AUGMENT = True
_C.TRAIN.BEST_SAVE_PATH = "best.keras"

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
