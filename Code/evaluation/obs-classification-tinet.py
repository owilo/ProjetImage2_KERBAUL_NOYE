import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image
import subprocess
import os
import random
import sys
import cv2
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import string
import time
from sklearn.metrics import confusion_matrix

if len(sys.argv) < 3:
    print("Format : python " + sys.argv[0] + " <Taille min de la région obscurcie> <Nombre de classes (<= 50)>")
    sys.exit()

class_count = int(sys.argv[2])
print("Chargement des données d'entraînement")
train_images = np.load("../datasets/tinet-train-dataset.npy")
train_labels = np.load("../datasets/tinet-train-labels.npy")

mask = train_labels < class_count
train_images = train_images[mask]
train_labels = train_labels[mask]

print("Chargement des données de test")
test_images = np.load("../datasets/tinet-test-dataset.npy")
test_labels = np.load("../datasets/tinet-test-labels.npy")

mask = test_labels < class_count
test_images = test_images[mask]
test_labels = test_labels[mask]

temp_dir = 'temp_test_images'
os.makedirs(temp_dir, exist_ok=True)

input_path = os.path.join(temp_dir, 'input.png')
output_path = os.path.join(temp_dir, 'output.png')

image_size = 64
if int(sys.argv[1]) > image_size:
    print("La taille de la région est invalide (" + sys.argv[2] + " > " + image_size + ")")
    sys.exit()
min_size_x = int(sys.argv[1])
min_size_y = int(sys.argv[1])

def generate_random_key(length=32):
    characters = string.ascii_letters + string.digits
    key = ''.join(random.choice(characters) for _ in range(length))
    return key

type_obs = 0

def apply_obs(image):
    global type_obs
    cv2.imwrite(input_path, image * 255)
    x1 = random.randint(0, image_size - min_size_x)
    y1 = random.randint(0, image_size - min_size_y)
    x2 = x1 + min_size_x
    y2 = y1 + min_size_y

    if type_obs == 0:
        return image, 0
    elif type_obs == 1:
        obs_size = random.randint(1, 4) * 2 + 1
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/floutage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif type_obs == 2:
        obs_size = random.randint(4, 16)
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/pixelisation", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif type_obs == 3:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        print("Params :", x1, y1, x2, y2, r, g, b)
        result = subprocess.run(
            ["../obscuration/masquage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(r), str(g), str(b)],
            capture_output=True, text=True
        )
    elif type_obs == 4:
        key = generate_random_key()
        print("Params :", x1, y1, x2, y2, key)
        result = subprocess.run(
            ["../obscuration/aes", input_path, output_path, str(x1), str(y1), str(x2), str(y2), key],
            capture_output=True, text=True
        )
    elif type_obs == 5:
        key = generate_random_key()
        nb = random.randint(3, 8)
        print("Params :", x1, y1, x2, y2, key, nb)
        result = subprocess.run(
            ["../obscuration/aes_bits", input_path, output_path, str(x1), str(y1), str(x2), str(y2), key, str(nb)],
            capture_output=True, text=True
        )
    elif type_obs == 6:
        dR = random.randint(5, 10) * random.choice([-1, 1])
        dG = random.randint(5, 10) * random.choice([-1, 1])
        dB = random.randint(5, 10) * random.choice([-1, 1])
        print("Params :", x1, y1, x2, y2, dR, dG, dB)
        result = subprocess.run(
            ["../obscuration/distorsion_RGB", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(dR), str(dG), str(dB)],
            capture_output=True, text=True
        )
    elif type_obs == 7:
        amp = random.uniform(1, 20)
        freq = random.uniform(0.1, 10)
        sens = random.choice([0, 1])
        print("Params :", x1, y1, x2, y2, amp, freq, sens)
        result = subprocess.run(
            ["../obscuration/distorsion_sinus", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(amp), str(freq), str(sens)],
            capture_output=True, text=True
        )
        

    obs_image = cv2.imread(output_path)

    if obs_image is None:
        print("Failed")
        return image, 0

    obs_image = obs_image / 255.0

    return obs_image, type_obs

final_train_images = []
final_train_labels = []

num_train_images = len(train_images)

for i in range(num_train_images):
    print(str(i + 1) + "/" + str(num_train_images))
    obs_image, obs_label = apply_obs(train_images[i])
    final_train_images.append(obs_image)
    final_train_labels.append(obs_label)
    type_obs = (type_obs + 1) % 8

final_train_images = np.array(final_train_images)
final_train_labels = np.array(final_train_labels)

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # 32
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # 16
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # 8
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

final_test_images = []
final_test_labels = []

num_test_images = len(test_images)

for i in range(num_test_images):
    print(str(i + 1) + "/" + str(num_test_images))
    obs_image, obs_label = apply_obs(train_images[i])
    final_test_images.append(obs_image)
    final_test_labels.append(obs_label)
    type_obs = (type_obs + 1) % 8

final_test_images = np.array(final_test_images)
final_test_labels = np.array(final_test_labels)

final_train_labels = to_categorical(final_train_labels, 8)
final_test_labels = to_categorical(final_test_labels, 8)

history = model.fit(
    final_train_images, final_train_labels,
    epochs=30,
    batch_size=4,
    validation_data=(final_test_images, final_test_labels),
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(final_test_images, final_test_labels)
shutil.rmtree(temp_dir)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Test accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_over_epochs.png')

predictions = model.predict(final_test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(final_test_labels, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)

types_obs = ["Aucun", "Floutage", "Masquage", "Pixél.", "AES", "AES/bits", "Dist. RGB", "Dist. géom."]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=types_obs, yticklabels=types_obs)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')