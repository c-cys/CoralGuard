from local_model import model
import numpy as np
import os
from PIL import Image

file_list = os.listdir('assets/coral_located')
images = [] ; labels = []
for file_name in file_list:
    file_path = os.path.join(file_list, file_name)
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((128, 128))
        images.append(np.array(img))
    except Exception as e:
        print(f"{e}: {file_name} image loading Error")
y_pred = model.predict(images)
y_pred_classes = np.argmax(y_pred, axis=1)