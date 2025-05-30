# FOMO (Faster Objects, More Objects)
FOMO is an object detection model developed by [Edge Impulse](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices) which is based on MobileNetV2.
However, this repo will provide Squeezenet, MobilenetV3 and MobileViT variants as well as MobileNetV2.

## Installation
`git clone git@github.com:bhoke/FOMO.git`

## Usage
For Tensorflow implementation, your custom dataloader should consume the dataset accordingly.

or you should write your own data loader class which can be eithe keras dataloader or `tf.data.Dataset` class.

When data loader is ready, you can train your model using:
`python train.py`

Finally, you can run inference on any image using:

`python predict.py`

## Supported Models
- [x] MobileNetV2
- [ ] Squeezenet
- [ ] MobilenetV3
- [ ] MobileViT

## Performance
Precision and hardware-specific metrics will be added here soon.
