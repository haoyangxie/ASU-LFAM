# Hybrid-CNN-LSTM
This code is about the paper "Online Thermal Profile Prediction for Large Format Additive Manufacturing: A Hybrid CNN-LSTM based Approach", which is currently under review of the AM jounal. 

Input data: 256*320 csv files. Model is tested to train on Tensorflow using GPU A100 with the IR thermal data for Table and Totem geometry. The model can be trained with following steps:

1. First, run **`CNN_pretrain.py`** and save the model. This step applies CNN autoencoder for spatial feature extraction in image. Three models including encoder, autoencoder, and decoder will be saved for later use. 
2. Second, import three saved CNN models and train the **`CNN_LSTM.py`**. After this step, the trained hybrid CNN-LSTM prediction model and inference prediction on test data will be saved.
3. Possible hyperparameter tuning in **`CNN_LSTM.py`**: number of past frames, number of future frames, learning rate, epochs, batch size.
<img width="1428" alt="CNN_LSTM_architecture" src="https://github.com/user-attachments/assets/41e36c07-1609-499a-9477-506d9962a216" />
