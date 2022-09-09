import json
import numpy as np

def bbox2segm(dataset, input_shape,
        output_width_height, num_classes_with_background):
    
    num_samples = len(dataset)
    out_in_ratio = input_shape[0] / output_width_height[0] # = 320 / 40 = 8 for classic FOMO
    out_shape = output_width_height + (num_classes_with_background,)
    with open("data/ams_labels.json") as lbl_file:
        ams_labels = json.load(lbl_file)

    segm_labels = np.zeros((num_samples, ) + out_shape)
    
    for sample_idx, sample in enumerate(ams_labels['samples'][:-5]):
        for bb in sample['boundingBoxes']:
            label = bb['label']
            leftmost = int(bb['x'] // out_in_ratio)
            top = int(bb['y'] // out_in_ratio)
            rightmost = int(leftmost + bb['w'] // out_in_ratio)
            bottom = int(top + bb['h'] // out_in_ratio)
            segm_labels[sample_idx, top:bottom, leftmost:rightmost, label] = 1
    
    # Subtract background
    segm_labels[:,:,:,0] = 1 - np.sum(segm_labels, 3)
    return segm_labels

    