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
# if apply two gpu
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


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


# ######################### load trained CNN model and encode to extract spatial features ###########################
autoencoder = load_model('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/autoencoder_3min.h5')
encoder = load_model('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/encoder_3min.h5')
decoder = load_model('/scratch/lliu112/CNN_LSTM/save_model/pretrained_CNN/decoder_3min.h5')
# evaluation = autoencoder.evaluate(test_CNN, test_CNN, verbose=2)
# print("CNN loss:", evaluation)

# generate feature map of last layer in encoding, shape=(32,40,32)
encoder_output = encoder.predict(frames_input)
print(encoder_output.shape)

lstm_input = pd.DataFrame(encoder_output.copy())

# ########################################## LSTM Training Model #################################################
######### define how many data for training, validation and testing
num_train = 4200     
train_input = lstm_input.iloc[:num_train, :]
test_input = lstm_input.iloc[num_train:, :]
# plt.plot(train_input.iloc[:, 21614])


######## define x and y
# split x and y by shifting
def lookback_split(data, n_past, n_future, gap):
    x_data = []
    y_data = []
    # n_past = 5
    # n_future = 1
    # data = training.copy()
    for i in range(n_past, len(data) - n_future - gap + 1):
        x_data.append(data.iloc[i - n_past:i, :])
        y_data.append(data.iloc[i + gap : i + gap + n_future, :])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


# Note: (samples, lookback, img_h, img_w, feature)
n_future = 15   # Number of days we want to look into the future based on the past days
n_past = 10  # Number of past days we want to use to predict the future
gap = 0 # timestamp to skip between past and future 
train_X, train_Y = lookback_split(train_input, n_past, n_future, gap)
test_X, test_Y = lookback_split(test_input, n_past, n_future, gap)

print('trainX shape == {}.'.format(train_X.shape))
print('trainY shape == {}.'.format(train_Y.shape))
print('testX shape == {}.'.format(test_X.shape))
print('testY shape == {}.'.format(test_Y.shape))


start_time_lstm_train = time.time()
######## define the LSTM Autoencoder model
# encoding and decoding type architecture
lstm_model = Sequential()
lstm_model.add(LSTM(4096, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='relu'))
# lstm_model.add(Dropout(0.25))
lstm_model.add(LSTM(2048, return_sequences=True, activation='relu'))
# lstm_model.add(Dropout(0.25))
lstm_model.add(LSTM(1024, return_sequences=False, activation='relu'))
# lstm_model.add(Dropout(0.2))
lstm_model.add(RepeatVector(train_Y.shape[1]))
lstm_model.add(LSTM(1024,  return_sequences=True, activation='relu'))
# lstm_model.add(Dropout(0.25))
lstm_model.add(LSTM(2048, return_sequences=True, activation='relu'))
# lstm_model.add(Dropout(0.25))
lstm_model.add(LSTM(4096, return_sequences=True, activation='relu'))
# lstm_model.add(Dropout(0.25))
lstm_model.add(TimeDistributed(Dense(train_Y.shape[2], activation='relu')))

optimizer = keras.optimizers.Adam(lr=1e-5)
lstm_model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
lstm_model.summary()

########## fit LSAM model
# fit the model
# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=35, mode='min')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=25, min_lr=1e-7, mode='min')
model_name = "/scratch/lliu112/CNN_LSTM/save_model/hybrid_prediction/temp_prediction_3min.h5"
if os.path.exists(model_name):
  os.remove(model_name)
else:
  print("The file does not exist")
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
history = lstm_model.fit(train_X, train_Y, epochs=350, batch_size=16, validation_data=(test_X, test_Y), verbose=2, callbacks=[early_stopping, reduce_lr, mcp_save])


end_time_lstm_train = time.time()
lstm_training_time = end_time_lstm_train - start_time_lstm_train
print("lstm training time: ", lstm_training_time)


# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()



##################################### LSTM Model Prediction on Testing Data #######################################
start_time_lstm_test = time.time()

lstm_model = load_model('/scratch/lliu112/CNN_LSTM/save_model/hybrid_prediction/temp_prediction_3min.h5')
prediction = lstm_model.predict(test_X)

end_time_lstm_test = time.time()
lstm_inference_time = end_time_lstm_test - start_time_lstm_test


########################## save frames for temperature compare ######################
import os, shutil
# clear directory files (raw image)
directory_list = ['/scratch/lliu112/CNN_LSTM/result/test_y_true',
                    '/scratch/lliu112/CNN_LSTM/result/test_y_pred'
                    ]


for directory in directory_list:
    folder = directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



train_frame = frames_input[:num_train, :, :, :]
test_frame = frames_input[num_train:, :, :, :]


# split x and y by shifting
def lookback_split_np(data, n_past, n_future, gap):
    x_data = []
    y_data = []
    for i in range(n_past, len(data) - n_future - gap + 1):
        x_data.append(data[i - n_past:i, :,:,:])
        y_data.append(data[i + gap : i + gap + n_future, :,:,:])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

# Note: (samples, lookback, img_h, img_w, feature)
x_train, y_train = lookback_split_np(train_frame, n_past, n_future, gap)
x_test, y_test = lookback_split_np(test_frame, n_past, n_future, gap)

plt.imshow(y_train[1][0])
plt.imshow(x_train[1][0])

print("x_train shape: ", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape:", y_test.shape)


def save_as_frame (index, n_future, data, path):
    # data shape: (n_future, 256, 320, 1)
    if index == 0:
        for i in np.arange(data.shape[0]):
            check = data[i]
            check = check.reshape(-1, check.shape[1])
            check = pd.DataFrame(check*255)
            check.to_csv(path + str(i) + ".csv", header=False, index=False)
    else:
        # assume gap = 0 
        check = data[-1] # get the last frame
        check = check.reshape(-1, check.shape[1])
        check = pd.DataFrame(check*255)
        check.to_csv(path + str(index-1+n_future) + ".csv", header=False, index=False)


### save y_test_true (the ground truth)
for j in np.arange(y_test.shape[0]):
    save_as_frame (j, n_future, y_test[j], "/scratch/lliu112/CNN_LSTM/result/test_y_true/test_y_true_")



### save y_test_pred (the prediction)
for j in np.arange(prediction.shape[0]):
    decoded_imgs = decoder.predict(prediction[j])
    save_as_frame (j, n_future, decoded_imgs, "/scratch/lliu112/CNN_LSTM/result/test_y_pred/test_y_pred_")

