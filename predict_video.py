from time import time_ns
import numpy as np
import cv2
from keras.saving import load_model
from matplotlib import pyplot as plt
from keras.applications import MobileNetV2


model_path = "virat_mobilenetv2.keras"
model = load_model(model_path)
vid_path = "samples/vtest.avi"
cv2.namedWindow("win", cv2.WINDOW_FREERATIO)


def img2tiles(image: np.ndarray):
    # tiles config
    tiles_height = 160
    tiles_width = 160
    num_y = int(image.shape[0] / tiles_height)
    num_x = int(image.shape[1] / tiles_width)
    new_width = num_x * tiles_width
    new_height = num_y * tiles_height
    new_img = np.zeros((num_x * num_y, tiles_height, tiles_width, 3))

    for incre_i, i in enumerate(range(0, new_height, tiles_height)):
        for incre_j, j in enumerate(range(0, new_width, tiles_width)):
            windx = incre_j + incre_i * num_x
            new_img[windx, ...] = image[i : i + tiles_height, j : j + tiles_width, :]

    return new_img, num_x, num_y


cap = cv2.VideoCapture(vid_path)
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
    # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    # height, width, _ = frame.shape
    # hpad = 0
    # vpad = 0

    # if width < height:
    #     new_width = round(240 * width / height)
    #     new_height = 240
    #     hpad = (new_height - new_width) // 2
    # else:
    #     new_width = 240
    #     new_height = round(240 * height / width)
    #     vpad = (new_width - new_height) // 2

    # cropped_frame = cv2.resize(frame[:,:,::-1], (new_width, new_height))
    # cropped_frame = np.pad(cropped_frame, ((vpad, vpad), (hpad, hpad), (0,0)))
    # height_ratio = height / new_height
    # width_ratio = width / new_width
    img, tiles_x, tiles_y = img2tiles(frame[..., ::-1])

    img = img.astype(np.float32) / 255.0
    # result = model(np.expand_dims(img, 0))
    start = time_ns()
    result = model(img)
    end = time_ns()
    print("Model inference time: ", (end - start) / 1e+6)
    num_classes = result.shape[3]
    for windx in range(result.shape[0]):
        for ch in range(1, num_classes):
            class_mask = result[windx, :, :, ch]
            binary_label = class_mask > 0.1
            y_coords, x_coords = np.where(binary_label)
            tile_row = windx // tiles_x
            tile_col = windx % tiles_x
            x_coords = x_coords * 8 + 160 * tile_col
            y_coords = y_coords * 8 + 160 * tile_row
            for x, y in zip(x_coords, y_coords):
                cv2.circle(frame, (x, y), 11, (0, 255, 0), -1)

    cv2.imshow("win", frame)
    cv2.waitKey(1)

cap.release()
