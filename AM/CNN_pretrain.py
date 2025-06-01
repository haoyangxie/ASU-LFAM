import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Lambda, Reshape
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import segmentation_models as sm
import time
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


################################ import frames
image_size_H = 256
image_size_W = 320
frames_input = []


for frame in np.arange(50,4600):
    # load temperature profile
    img_thermal = pd.read_csv(
        "/scratch/lliu112/CNN_LSTM/data/hex3min/hex_3min_" + str(
            frame) + ".csv", header=None)
            
    geometry = np.expand_dims(img_thermal, axis = -1) # add dimension of the color channel since 2D temperature profile
    frames_input.append(geometry)

# normalize pixel values
frames_input = np.array(frames_input).astype('float32')/255
print("frame_input shape:", frames_input.shape)


# ###################### CNN Encoding & Decoding Model ############################
train_CNN = frames_input[:4200, :, :, :]
test_CNN = frames_input[4200:, :, :, :]

input_img = Input(shape=(256, 320, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

bottleneck = Dense(4096, activation='relu')(x)

# at this point the representation is (8,10,512) 
x = Dense(8*10*512, activation='relu')(bottleneck)  
x = Reshape((8,10,512))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)

decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)


autoencoder = Model(input_img, decoded)
optimizer = keras.optimizers.Adam(lr=1e-5)
autoencoder.compile(optimizer = optimizer, loss='mse', metrics=['mae'])
autoencoder.summary()


early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=20, min_lr=1e-7)
hist2 = autoencoder.fit(train_CNN, train_CNN,
                epochs=200,
                batch_size=32,
                verbose = 2,
                validation_data=(test_CNN, test_CNN),callbacks=[early_stopping, reduce_lr])


plt.plot(hist2.history['loss'], label='Training loss')
plt.plot(hist2.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()



# define the encoder part
encoder = Model(input_img, bottleneck)
# encoder.summary()

# degine the decoder part
bottleneck_input = autoencoder.layers[-22].output
decoder_layer = autoencoder.layers[-21](bottleneck_input)
decoder_layer = autoencoder.layers[-20](decoder_layer)
decoder_layer = autoencoder.layers[-19](decoder_layer)
decoder_layer = autoencoder.layers[-18](decoder_layer)
decoder_layer = autoencoder.layers[-17](decoder_layer)
decoder_layer = autoencoder.layers[-16](decoder_layer)
decoder_layer = autoencoder.layers[-15](decoder_layer)
decoder_layer = autoencoder.layers[-14](decoder_layer)
decoder_layer = autoencoder.layers[-13](decoder_layer)
decoder_layer = autoencoder.layers[-12](decoder_layer)
decoder_layer = autoencoder.layers[-11](decoder_layer)
decoder_layer = autoencoder.layers[-10](decoder_layer)
decoder_layer = autoencoder.layers[-9](decoder_layer)
decoder_layer = autoencoder.layers[-8](decoder_layer)
decoder_layer = autoencoder.layers[-7](decoder_layer)
decoder_layer = autoencoder.layers[-6](decoder_layer)
decoder_layer = autoencoder.layers[-5](decoder_layer)
decoder_layer = autoencoder.layers[-4](decoder_layer)
decoder_layer = autoencoder.layers[-3](decoder_layer)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(bottleneck_input, decoder_layer)
# decoder.summary()


# save trained CNN model
autoencoder.save('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/autoencoder_3min.h5')
encoder.save('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/encoder_3min.h5')
decoder.save('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/decoder_3min.h5')


