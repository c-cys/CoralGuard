import kagglehub
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import zipfile

bleached = kagglehub.dataset_download("cchoiys/coralguard-bleached")
normal = kagglehub.dataset_download("cchoiys/coralguard-normal")
print(bleached, normal, sep='\n')

bleached_label = 1
normal_label = 0

images = []
labels = []

def load_images_from_folder(folder, label):
    if label == 1:
        global bleached_file_names
        bleached_file_names = os.listdir(folder)
        file_names = bleached_file_names
        print(bleached_file_names)
    if label == 0:
        global normal_file_names
        normal_file_names = os.listdir(folder)
        file_names = normal_file_names
        print(normal_file_names)

    for filename in file_names:
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((128, 128))
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"{e}: {filename} image loading Error")

load_images_from_folder(bleached, bleached_label)
load_images_from_folder(normal, normal_label)

images = np.array(images)
labels = np.array(labels)

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.4, random_state=42
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42
)

print("Train Images:", train_images.shape)
print("Validation Images:", val_images.shape)
print("Test Images:", test_images.shape)

output_zip_path = "assets/coral_dataset.zip"

with zipfile.ZipFile(output_zip_path, 'w') as zipf:
    for filename in bleached_file_names:
        file_path = os.path.join(bleached, filename)
        img = Image.open(file_path).resize((128, 128))
        temp_path = f"assets/dataset/bleached/{filename}"
        img.save(temp_path)
        zipf.write(temp_path, arcname=temp_path)
        os.remove(temp_path)

    for filename in normal_file_names:
        file_path = os.path.join(normal, filename)
        img = Image.open(file_path).resize((128, 128))
        temp_path = f"assets/dataset/normal/{filename}"
        img.save(temp_path)
        zipf.write(temp_path, arcname=temp_path)
        os.remove(temp_path)