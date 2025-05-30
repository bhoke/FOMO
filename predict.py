import numpy as np
from PIL import Image
from keras.saving import load_model
from matplotlib import pyplot as plt

best_model_path = "best.keras"
model = load_model(best_model_path)
im_path = "dataset/test/9edited_192.jpg"
img = np.array(Image.open(im_path).resize((224,224)), dtype=np.float32)/255.0
result = model(np.expand_dims(img, 0))
fig, (ax1,ax2) = plt.subplots(1,2)
print(img.shape)
print(result.shape)
ax1.imshow(img)
ax2.imshow(result[0,...,1])
plt.show()
