import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def array_to_pil_image(img_array):
    img_array = np.squeeze(img_array, axis=0)
    img_array = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    return pil_img

if len(sys.argv) < 8:
    print("Format : python " + sys.argv[0] + " <Image d'entrée> <Image de sortie> <x1> <y1> <x2> <y2> <Nombre de caractéristiques obscurées (0 - 2048)>")
    sys.exit(1)

img_path = sys.argv[1]
output_path = sys.argv[2]

encoder = load_model('lfw-encoder.keras')
decoder = load_model('lfw-decoder.keras')

input_img = preprocess_image(img_path)

latent = encoder.predict(input_img)
import numpy as np

n = int(sys.argv[7])

all_indices = [(i, j, k) for i in range(4) for j in range(4) for k in range(128)]

np.random.shuffle(all_indices)

selected_indices = all_indices[:n]

for i, j, k in selected_indices:
    latent[0, i, j, k] = np.random.rand()


output_img = decoder.predict(latent)

output_img = np.squeeze(output_img)
output_img = np.clip(output_img * 255, 0, 255).astype('uint8')
result_img = array_to_img(output_img)

crop_box = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
cropped_section = result_img.crop(crop_box)
res = array_to_pil_image(input_img)
res.paste(cropped_section, crop_box)
res.save(output_path)
