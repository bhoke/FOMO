import numpy as np
from PIL import Image
from keras.saving import load_model
from matplotlib import pyplot as plt

best_model_path = "model.h5"
model = load_model(best_model_path)
im_path = "/home/berko/Downloads/Untitled.jpg"
img = np.array(Image.open(im_path), dtype=np.float32)/255.0
result = model(np.expand_dims(img, 0))
fig, (ax1,ax2) = plt.subplots(1,2)
print(img.shape)
print(result.shape)
ax1.imshow(img)
ax2.imshow(result[0,...,1])
plt.show()
