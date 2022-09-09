import tensorflow as tf

def DiceLoss(gt, preds, smooth=1e-6):
    gt = gt[...,1:]
    preds = preds[...,1:]
    reduce_axes = range(1, len(preds.shape))
    intersection = tf.reduce_sum(gt * preds, reduce_axes) # batch_size
    dice = (2*intersection + smooth) / (tf.reduce_sum(gt, reduce_axes) + tf.reduce_sum(preds, reduce_axes) + smooth)
    return 1 - dice

