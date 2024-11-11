import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import subprocess
import os
import random
import sys
import cv2
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import confusion_matrix

if len(sys.argv) < 3:
    print("Format : python " + sys.argv[0] + " <Nom de la méthode d'obscuration> <Taille min de la région obscurcie>")
    sys.exit()

(train_images, _), (test_images, _) = datasets.cifar10.load_data()

train_images = train_images[:25000]

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

temp_dir = 'temp_test_images'
os.makedirs(temp_dir, exist_ok=True)

input_path = os.path.join(temp_dir, 'input.png')
output_path = os.path.join(temp_dir, 'output.png')

image_size = 32
if int(sys.argv[2]) > image_size:
    print("La taille de la région est invalide (" + sys.argv[2] + " > " + image_size + ")")
    sys.exit()
min_size_x = int(sys.argv[2])
min_size_y = int(sys.argv[2])

def generate_random_key(length=32):
    characters = string.ascii_letters + string.digits
    key = ''.join(random.choice(characters) for _ in range(length))
    return key

def apply_obs(image):
    cv2.imwrite(input_path, image * 255)
    x1 = random.randint(0, image_size - min_size_x)
    y1 = random.randint(0, image_size - min_size_y)
    x2 = x1 + min_size_x
    y2 = y1 + min_size_y

    if sys.argv[1] == "floutage":
        obs_size = random.randint(1, 4) * 2 + 1
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/floutage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "pixelisation":
        obs_size = random.randint(4, 16)
        print("Params :", x1, y1, x2, y2, obs_size)
        result = subprocess.run(
            ["../obscuration/pixelisation", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(obs_size)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "masquage":
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        print("Params :", x1, y1, x2, y2, r, g, b)
        result = subprocess.run(
            ["../obscuration/masquage", input_path, output_path, str(x1), str(y1), str(x2), str(y2), str(r), str(g), str(b)],
            capture_output=True, text=True
        )
    elif sys.argv[1] == "aes":
        key = generate_random_key()
        print("Params :", x1, y1, x2, y2, key)
        result = subprocess.run(
            ["../obscuration/aes", input_path, output_path, str(x1), str(y1), str(x2), str(y2), key],
            capture_output=True, text=True
        )
    else:
        print("Méthode inconnue, stop")
        sys.exit()

    if not os.path.exists(output_path):
        return image, 0

    obs_image = cv2.imread(output_path)
    obs_image = obs_image / 255.0

    return obs_image, 1

final_train_images = train_images
train_labels = []

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
    obs_image, obs_label = apply_obs(train_images[i])
    obs_train_images.append(obs_image)
    train_labels.append(obs_label)

for i in unobs_indices:
    unobs_train_images.append(train_images[i])
    train_labels.append(0)

final_train_images = np.array(obs_train_images + unobs_train_images)
final_train_labels = np.array(train_labels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

indices = list(range(len(test_images)))
random.shuffle(indices)
obs_indices = indices[:len(test_images) // 2]
unobs_indices = indices[len(test_images) // 2:]

final_test_images = []
final_test_labels = []

j = 0
for i in obs_indices:
    j += 1
    print(str(j) + "/" + str(len(obs_indices)))
    img, label = apply_obs(test_images[i])
    final_test_images.append(img)
    final_test_labels.append(label)

for i in unobs_indices:
    final_test_images.append(test_images[i])
    final_test_labels.append(0)

final_test_images = np.array(final_test_images)
final_test_labels = np.array(final_test_labels)

history = model.fit(
    final_train_images, final_train_labels,
    epochs=30,
    batch_size=64,
    validation_data=(final_test_images, final_test_labels),
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(final_test_images, final_test_labels)
shutil.rmtree(temp_dir)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Test accuracy')
plt.title('Offuscation detection accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('offuscation_accuracy_over_epochs.png')

predictions = model.predict(final_test_images)
predicted_offuscation = (predictions > 0.5).astype(int)

cm_offuscation = confusion_matrix(final_test_labels, predicted_offuscation, labels=[0, 1])
plt.figure(figsize=(10, 8))
sns.heatmap(cm_offuscation, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Offusquée', 'Offusquée'], yticklabels=['Non Offusquée', 'Offusquée'])
plt.title('Confusion Matrix - Offuscation Detection')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_offuscation.png')