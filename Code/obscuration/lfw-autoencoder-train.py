"""import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model

images = np.load("../datasets/lfw-dataset.npy")

x_train, x_test = train_test_split(images, test_size=0.1, random_state=0)

input_img = Input(shape=(256, 256, 3))

# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=8,
    shuffle=True,
    validation_data=(x_test, x_test)
)

autoencoder.save('lfw-autoencoder.keras')

encoder = Model(inputs=input_img, outputs=encoded)
encoder.save('lfw-encoder.keras')

encoded_input = Input(shape=(32, 32, 128))
decoder_layer = autoencoder.layers[-7](encoded_input)
for layer in autoencoder.layers[-6:]:
    decoder_layer = layer(decoder_layer)
decoder = Model(inputs=encoded_input, outputs=decoder_layer)
decoder.save('lfw-decoder.keras')"""


import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

images = np.load("../datasets/lfw-dataset.npy")

x_train, x_test = train_test_split(images, test_size=0.1, random_state=0)

input_shape = (256, 256, 3)

# Encoder
def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Flatten()(x)
    
    latent = layers.Dense(64, activation='relu')(x)
    
    encoder = models.Model(inputs, latent, name="encoder")
    return encoder

# Decoder
def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    x = layers.Dense(32 * 32 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((32, 32, 64))(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    
    decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = models.Model(latent_inputs, decoded, name="decoder")
    return decoder

encoder = build_encoder(input_shape)
decoder = build_decoder(64)

autoencoder_input = layers.Input(shape=input_shape)
latent_representation = encoder(autoencoder_input)
reconstructed_image = decoder(latent_representation)

autoencoder = models.Model(autoencoder_input, reconstructed_image, name="autoencoder")

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=8,
    shuffle=True,
    validation_data=(x_test, x_test)
)

encoder.save('lfw-encoder.keras')
decoder.save('lfw-decoder.keras')