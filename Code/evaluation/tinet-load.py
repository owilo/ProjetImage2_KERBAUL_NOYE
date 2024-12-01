import os
import numpy as np
import sys
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if len(sys.argv) < 2:
    print("Format : python " + sys.argv[0] + " <Nombre de classes max> <Nombre d'images max>")
    sys.exit(1)

train_images = []
train_labels = []
test_images = []
test_labels = []
max_class = int(sys.argv[1])
max_img = int(sys.argv[2])

i = 1
train_data_dir = "/mnt/c/Users/Valentin/Documents/temp/archive/tiny-imagenet-200/tiny-imagenet-200/train"
ddir = os.listdir(train_data_dir)
random.shuffle(ddir)
for clazz in ddir:
    if i > max_class:
        break

    train_class_path = os.path.join(train_data_dir, clazz, "images")
    print("Chargement du dossier", train_class_path, i, "/", min(max_class, len(ddir)))
    if not os.path.isdir(train_class_path):
        continue
    
    j = 0
    class_images = []  # Liste pour stocker les images de la classe
    idir = os.listdir(train_class_path)
    for image_name in idir:
        if j >= max_img:
            break

        image_path = os.path.join(train_class_path, image_name)
        
        img = Image.open(image_path).convert('RGB')
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        
        if j < 0.8 * max_img:
            train_images.append(img_array)
            train_labels.append(i - 1)
        else:
            test_images.append(img_array)
            test_labels.append(i - 1)

        class_images.append(img_array)

        j += 1
        print(j, "/", min(max_img, len(idir)))

    if max_img > 500:
        print(f"Augmentation des données pour la classe {clazz}, nombre d'images: {len(class_images)}")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        class_images = np.array(class_images)

        datagen.fit(class_images)

        total_needed = max_img - len(class_images)
        augmented_images = []

        for batch in datagen.flow(class_images, batch_size=1, save_to_dir=None, shuffle=True):
            augmented_image = np.asarray(batch[0], dtype=np.float32) / 255.0

            augmented_images.append(augmented_image)
            train_labels.append(i - 1)
            
            if len(augmented_images) >= total_needed:
                break

        augmented_images = np.array(augmented_images)
        train_images += augmented_images.tolist()

        print("Nombre d'images total :", len(train_images))

    print("Fin du chargement du dossier", i, "/", min(max_class, len(ddir)))
    i += 1

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print("Sauvegarde...")
np.save("../datasets/tinet-train-dataset.npy", train_images)
print("Sauvegarde 1/4")
np.save("../datasets/tinet-train-labels.npy", train_labels)
print("Sauvegarde 2/4")
np.save("../datasets/tinet-test-dataset.npy", test_images)
print("Sauvegarde 3/4")
np.save("../datasets/tinet-test-labels.npy", test_labels)
print("Sauvegarde 4/4")