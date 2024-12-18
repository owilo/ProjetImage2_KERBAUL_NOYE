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
import cv2
import sys
import random
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import confusion_matrix

if len(sys.argv) < 5:
    print("Format : python " + sys.argv[0] + " <Nom de la méthode d'obscuration> <Obscurer le dataset d'entraînement? (0/1)> <Taille min de la région obscurcie> <Nombre de classes (<= 50)")
    sys.exit()

class_count = int(sys.argv[4])

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

print("Fin du chargement")

#train_images = train_images.astype('float32') / 255.0
#test_images = test_images.astype('float32') / 255.0

train_labels = to_categorical(train_labels, class_count)
test_labels = to_categorical(test_labels, class_count)

temp_dir = 'temp_test_images'
os.makedirs(temp_dir, exist_ok=True)

input_path = os.path.join(temp_dir, 'input.png')
output_path = os.path.join(temp_dir, 'output.png')

image_size = 64
if int(sys.argv[3]) > image_size:
    print("La taille de la région est invalide (" + sys.argv[3] + " > " + image_size + ")")
    sys.exit()
min_size_x = int(sys.argv[3])
min_size_y = int(sys.argv[3])

def generate_random_key(length=32):
    characters = string.ascii_letters + string.digits
    key = ''.join(random.choice(characters) for _ in range(length))
    return key

type_obs = 0

def apply_obs(image):
    global type_obs
    cv2.imwrite(input_path, image * 255)

    x1 = random.randint(0, image_size - min_size_x) if min_size_x != image_size else 0
    y1 = random.randint(0, image_size - min_size_y) if min_size_y != image_size else 0
    x2 = x1 + min_size_x
    y2 = y1 + min_size_y

    if sys.argv[1] == "none":
        return image
    if sys.argv[1] == "floutage" or sys.argv[1] == "all" and type_obs == 0:
        obs_size = random.randint(1, 4) * 2 + 1
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/floutage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "pixelisation" or sys.argv[1] == "all" and type_obs == 1:
        obs_size = random.randint(4, 16)
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/pixelisation", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "masquage" or sys.argv[1] == "all" and type_obs == 2:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        print("Params :", x1, y1, x2, y2, r, g, b)
        result = subprocess.run(
            ["../obscuration/masquage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(r), str(g), str(b)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "aes" or sys.argv[1] == "all" and type_obs == 3:
        key = generate_random_key()
        print("Params :", x1, y1, x2, y2, key)
        result = subprocess.run(
            ["../obscuration/aes", input_path, output_path, str(x1), str(y1), str(x2), str(y2), key],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "aes_bits" or sys.argv[1] == "all" and type_obs == 4:
        key = generate_random_key()
        nb = random.randint(3, 8)
        print("Params :", x1, y1, x2, y2, key, nb)
        result = subprocess.run(
            ["../obscuration/aes_bits", input_path, output_path, str(x1), str(y1), str(x2), str(y2), key, str(nb)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "distorsion_RGB" or sys.argv[1] == "all" and type_obs == 5:
        dR = random.randint(5, 10) * random.choice([-1, 1])
        dG = random.randint(5, 10) * random.choice([-1, 1])
        dB = random.randint(5, 10) * random.choice([-1, 1])
        print("Params :", x1, y1, x2, y2, dR, dG, dB)
        result = subprocess.run(
            ["../obscuration/distorsion_RGB", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(dR), str(dG), str(dB)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "distorsion_sinus" or sys.argv[1] == "all" and type_obs == 6:
        amp = random.uniform(1, 20)
        freq = random.uniform(0.1, 10)
        sens = random.choice([0, 1])
        print("Params :", x1, y1, x2, y2, amp, freq, sens)
        result = subprocess.run(
            ["../obscuration/distorsion_sinus", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(amp), str(freq), str(sens)],
            capture_output=True, text=True
        )
    else:
        print("Méthode inconnue, stop")
        sys.exit()

    type_obs = (type_obs + 1) % 7

    """if result.returncode != 0:
        print(f"Erreur pendant l'appel au programme: {result.stderr}")
        return image"""

    if not os.path.exists(output_path):
        print(f"Chemin {output_path} inconnu")
        return image

    obs_image = cv2.imread(output_path)
    obs_image = cv2.resize(obs_image, (image_size, image_size))
    if obs_image is None:
        print(f"L'image de sortie {output_path} n'a pas pu être chargée")
        return image

    obs_image = obs_image / 255.0
    return obs_image

final_train_images = train_images
final_train_labels = train_labels
if sys.argv[2] == "1":
    print("Preprocessing des images d'entraînement")

    num_train_images = len(train_images)
    obs_train_images = []
    unobs_train_images = []

    indices = list(range(num_train_images))
    random.shuffle(indices)
    obs_indices = indices[:num_train_images // 2]
    unobs_indices = indices[num_train_images // 2:]

    j = 0
    for i in obs_indices:
        j += 1
        print(str(j) + "/" + str(len(obs_indices)))
        obs_train_images.append(apply_obs(train_images[i]))

    for i in unobs_indices:
        unobs_train_images.append(train_images[i])

    final_train_images = np.array(obs_train_images + unobs_train_images)
    final_train_labels = np.array([train_labels[i] for i in obs_indices] + [train_labels[i] for i in unobs_indices])

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
    layers.Dense(class_count, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

num_images = len(test_images)
obs_test_images = []
unobs_test_images = []

print("Preprocessing des images de test")

for i in range(num_images):
    print(str(i + 1) + "/" + str(num_images))
    obs_test_images.append(apply_obs(test_images[i]))

final_test_images = np.array(obs_test_images)
final_test_labels = np.array(test_labels)  

print("Fitting")

history = model.fit(
    final_train_images, final_train_labels,
    epochs=30,
    batch_size=16,
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

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')