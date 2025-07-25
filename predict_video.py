from time import time_ns
import numpy as np
import cv2
from keras.saving import load_model
from matplotlib import pyplot as plt
from keras.applications import MobileNetV2


model_path = "virat_mobilenetv2.keras"
model = load_model(model_path, compile=False)
vid_path = "samples/VIRAT_S_000200_06_001693_001824.mp4"
cv2.namedWindow("win", cv2.WINDOW_FREERATIO)


cap = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vidWriter = cv2.VideoWriter("demos/VIRAT_S_000200_06_001693_001824.mp4", fourcc, 30.0, (640, 360))
if not cap.isOpened():
    print("Cannot open video file")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("End of stream. Exiting ...")
        break

    height, width, _ = frame.shape
    hpad = 0
    vpad = 0

    if width < height:
        new_width = round(640 * width / height)
        new_height = 360
        hpad = (640 - new_width) // 2
    else:
        new_width = 640
        new_height = round(640 * height / width)
        vpad = (360 - new_height) // 2

    resized_frame = cv2.resize(frame[:, :, ::-1], (new_width, new_height))
    if vpad < 0:
        resized_frame = resized_frame[-vpad : new_height + vpad, ...]
        vpad = 0
    elif hpad < 0:
        resized_frame = resized_frame[:, -hpad : new_height + hpad, :]
        hpad = 0
    else:
        resized_frame = np.pad(resized_frame, ((vpad, vpad), (hpad, hpad), (0, 0)))

    img = resized_frame.astype(np.float32) / 255.0
    start = time_ns()
    result = model(np.expand_dims(img, 0))
    end = time_ns()
    print("Model inference time: ", (end - start) / 1e6)
    num_classes = result.shape[3]
    for ch in range(1, num_classes):
        color = [0, 0, 0]
        color[ch] = 255
        class_mask = result[0,..., ch]
        binary_label = (class_mask > 0.9).numpy().astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_label, None, None, None)
        center_idxs = np.where((stats[1:,4] > 0))[0]
        if (len(center_idxs) > 0):
            for x, y in centroids[center_idxs + 1]:
                cv2.circle(resized_frame, (int(x * 8 + hpad), int(y * 8 + vpad)), 7, color, 2)

    vidWriter.write(resized_frame)
    cv2.imshow("win", resized_frame[...,::-1])
    cv2.waitKey(1)

vidWriter.release()
cap.release()
cv2.destroyAllWindows()
