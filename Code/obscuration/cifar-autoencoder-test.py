import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if len(sys.argv) < 4:
    print("Format : python " + sys.argv[0] + " <Image d'entrée> <Image de sortie> <Nombre de caractéristiques obscurées (0 - 4)>")
    sys.exit(1)

img_path = sys.argv[1]
model_path = 'cifar-autoencoder.h5'

autoencoder = load_model(model_path)

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-8].output)
latent_dim = encoder.output_shape[-1]
decoder_input = tf.keras.layers.Input(shape=(4, 4, 64))
decoder_output = autoencoder.layers[-7](decoder_input)
for layer in autoencoder.layers[-6:]:
    decoder_output = layer(decoder_output)
decoder = Model(inputs=decoder_input, outputs=decoder_output)

input_img = preprocess_image(img_path)

latent = encoder.predict(input_img)

n = int(sys.argv[3])

random_indices = np.random.choice(latent.shape[1], size=n, replace=False)

for i in random_indices:
    latent[:, i] = np.random.rand()

output_img = decoder.predict(latent)

output_img = np.squeeze(output_img)
output_img = np.clip(output_img * 255, 0, 255).astype('uint8')
result_img = array_to_img(output_img)
result_img.save(sys.argv[2])