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

if len(sys.argv) < 4:
    print("Format : python " + sys.argv[0] + " <Nom de la méthode d'obscuration> <Obscurer le dataset d'entraînement? (0/1)> <Taille min de la région obscurcie>")
    sys.exit()

"""
--------
"""

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def array_to_pil_image(img_array):
    img_array = np.squeeze(img_array, axis=0)
    img_array = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    return pil_img

model_path = '../obscuration/cifar-autoencoder.keras'
autoencoder = load_model(model_path)

def ae(input, output, x1, y1, x2, y2, n):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-8].output)
    decoder_input = tf.keras.layers.Input(shape=(4, 4, 64))
    decoder_output = autoencoder.layers[-7](decoder_input)
    for layer in autoencoder.layers[-6:]:
        decoder_output = layer(decoder_output)
    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    input_img = preprocess_image(input)

    latent = encoder.predict(input_img)

    random_indices = np.random.choice(latent.shape[1], size=n, replace=False)

    for i in random_indices:
        latent[:, i] = np.random.rand()

    output_img = decoder.predict(latent)

    output_img = np.squeeze(output_img)
    output_img = np.clip(output_img * 255, 0, 255).astype('uint8')

    result_img = array_to_img(output_img)

    crop_box = (x1, y1, x2, y2)

    cropped_section = result_img.crop(crop_box)
    res = array_to_pil_image(input_img)
    res.paste(cropped_section, crop_box)
    res.save(output)

"""
--------
"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

if sys.argv[2] == "1":
    train_images = train_images[:25000]
    train_labels = train_labels[:25000]
    
if sys.argv[1] == "ae":
    if sys.argv[2] == "1":
        train_images = train_images[:500]
        train_labels = train_labels[:500]
    test_images = train_images[:100]
    test_labels = train_labels[:100]

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

temp_dir = 'temp_test_images'
os.makedirs(temp_dir, exist_ok=True)

input_path = os.path.join(temp_dir, 'input.png')
output_path = os.path.join(temp_dir, 'output.png')

image_size = 32
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
    elif sys.argv[1] == "ae" or sys.argv[1] == "all" and type_obs == 7:
        n = random.randint(0, 4)
        print("Params :", x1, y1, x2, y2, n)
        ae(input_path, output_path, x1, y1, x2, y2, n)
    else:
        print("Méthode inconnue, stop")
        sys.exit()

    type_obs = (type_obs + 1) % 8

    """if result.returncode != 0:
        print(f"Erreur pendant l'appel au programme: {result.stderr}")
        return image"""

    if not os.path.exists(output_path):
        print(f"Chemin {output_path} inconnu")
        return image

    obs_image = cv2.imread(output_path)
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
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
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
    batch_size=64,
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