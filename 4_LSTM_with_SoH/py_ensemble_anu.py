import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import os
import math

# Load dataset
dir = 'refined_dataset'
listdir = os.listdir(dir)

num = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
for i in range(len(listdir)):
    vector = np.zeros((1,3))
    path = os.path.join(os.getcwd(), 'refined_dataset/', num[i] + '_discharge_soh.csv')
    csv = pd.read_csv(path)
    df = pd.DataFrame(csv)
    vec = df[['cycle', 'capacity', 'SOH']]
    globals()['data_{}'.format(num[i])] = vec

data = pd.read_csv('refined_dataset/B05_discharge_soh.csv')
df = pd.DataFrame(data)

data_B05 = globals()['data_B05']

# Prepare dataset
dataset = data_B05["SOH"].values
cycle = data_B05['cycle'].values
dataset = dataset.reshape((len(dataset), 1))

# Split dataset
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

def build_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

n_boosting_rounds = 3
predictions = np.zeros_like(testY)

for round in range(n_boosting_rounds):
    model = build_model()
    model.fit(trainX, trainY, epochs=200, batch_size=20, validation_data=(testX, testY), verbose=1, shuffle=False)
    
    # Predict the residuals
    residuals = trainY - model.predict(trainX).flatten()
    trainY = residuals  # Update trainY to be the residuals for the next model

    # Aggregate the predictions
    predictions += model.predict(testX).flatten()

# Average the predictions over the boosting rounds
predictions /= n_boosting_rounds

# Plot predictions vs real data
plt.plot(predictions, label='Prediction')
plt.plot(testY, label='Real data')
plt.legend()
plt.show()

# Calculate RMSE and MAE
rmse = math.sqrt(mean_squared_error(testY, predictions))
mae = mean_absolute_error(testY, predictions)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
