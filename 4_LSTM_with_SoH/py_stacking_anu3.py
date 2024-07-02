import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# ## 1. Load Dataset

dir = 'refined_dataset'
listdir = os.listdir(dir)

print(listdir)
print("The number of dataset:", len(listdir))

num = ['B05', 'B07', 'B18', 'B33', 'B34', 'B46', 'B47', 'B48']
for i in range(len(listdir)):
    vector = np.zeros((1, 3))
    path = os.path.join(os.getcwd(), 'refined_dataset/', num[i] + '_discharge_soh.csv')
    csv = pd.read_csv(path)
    df = pd.DataFrame(csv)
    
    vec = df[['cycle', 'capacity', 'SOH']]
    
    globals()['data_{}'.format(num[i])] = vec

data = pd.read_csv('refined_dataset/B05_discharge_soh.csv')
df = pd.DataFrame(data)

# ## 2. Split train and test dataset

dataset = data["SOH"]
cycle = data['cycle']

dataset = np.array(dataset)
dataset = dataset.reshape((len(dataset), 1))

train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

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

# ## 3. Define and Tune the Model with Keras Tuner

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        input_shape=(trainX.shape[1], trainX.shape[2]),
        return_sequences=True
    ))
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32)
    ))
    model.add(Dense(1))
    
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='mae'
    )
    return model

# Initialize Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=1,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='soh_prediction'
)

# Search for the best hyperparameters
tuner.search(trainX, trainY, epochs=50, validation_data=(testX, testY))

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best Hyperparameters: {best_hps.values}")

# Train the best model
history = best_model.fit(trainX, trainY, epochs=50, batch_size=20, validation_data=(testX, testY), verbose=1)

# Save the best model
model_json = best_model.to_json()
with open('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/B48_best_model.json', 'w') as json_file:
    json_file.write(model_json)

# Correct filename extension for saving weights
best_model.save_weights('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/B48_best_weights.weights.h5')

# ## 4. Load Trained Best Model

from keras.models import model_from_json 

# Load the model structure
with open('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/B48_best_model.json', 'r') as json_file:
    loaded_model_json = json_file.read() 

loaded_model = model_from_json(loaded_model_json)

# Load model weights with the correct filename
loaded_model.load_weights('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/B48_best_weights.weights.h5')
print("Loaded best model from disk")


# ## 5. Evaluate the Best Model

yhat = loaded_model.predict(testX)
tyhat = loaded_model.predict(trainX)

cycle1 = cycle[0:train_size]
cycle2 = cycle[train_size:]
trainX_reshaped = trainX.reshape((trainX.shape[0], trainX.shape[2]))
yhat_reshaped = yhat.reshape((yhat.shape[0],))

sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

# Ensure cycle2, testY, and yhat_reshaped have consistent lengths
cycle2 = cycle2[:len(testY)]
yhat_reshaped = yhat_reshaped[:len(testY)]

plt.plot(cycle1[:-1], trainX_reshaped[:, 0], label='Used real data', linewidth=3, color='r')
plt.plot(cycle2, testY, label='Real data', linewidth=3, color='b')
plt.plot(cycle2, yhat_reshaped, label='LSTM Prediction', linewidth=3, color='g')
plt.legend(prop={'size': 16})

plt.ylabel('SoH', fontsize=15)
plt.xlabel('Discharge cycle', fontsize=15)
plt.title("SOH Prediction", fontsize=15)
plt.savefig('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/50%/fig/B48_LSTM_best.jpg')
plt.show()


# Print RMSE and MAE
rmse = math.sqrt(mean_squared_error(testY, yhat))
mae = mean_absolute_error(testY, yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)

# ## 6. Implement Stacking

# Define a function to create multiple LSTM models
def create_lstm_models(num_models, input_shape):
    models = []
    for _ in range(num_models):
        model = Sequential()
        model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(units=64))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')
        models.append(model)
    return models

# Train multiple LSTM models
num_models = 5
base_models = create_lstm_models(num_models, (trainX.shape[1], trainX.shape[2]))

# Train each base model
for model in base_models:
    model.fit(trainX, trainY, epochs=50, batch_size=20, validation_data=(testX, testY), verbose=1)

# Generate predictions from each base model
train_meta_input = np.zeros((trainX.shape[0], num_models))
test_meta_input = np.zeros((testX.shape[0], num_models))

for i, model in enumerate(base_models):
    train_meta_input[:, i] = model.predict(trainX).flatten()
    test_meta_input[:, i] = model.predict(testX).flatten()

# Define the meta-model
meta_model = Sequential()
meta_model.add(Dense(units=64, input_dim=num_models, activation='relu'))
meta_model.add(Dense(1))
meta_model.compile(optimizer='adam', loss='mae')

# Train the meta-model
meta_model.fit(train_meta_input, trainY, epochs=50, batch_size=20, validation_data=(test_meta_input, testY), verbose=1)

# Evaluate the stacked model
stacked_predictions = meta_model.predict(test_meta_input)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(cycle2, testY, label='Real data', linewidth=3, color='b')
plt.plot(cycle2, stacked_predictions, label='Stacked LSTM Prediction', linewidth=3, color='g')
plt.legend(prop={'size': 16})
plt.ylabel('SoH', fontsize=15)
plt.xlabel('Discharge cycle', fontsize=15)
plt.title("Stacked LSTM SOH Prediction", fontsize=15)
plt.savefig('D:/SOH_RUL_Estimation-main/4_LSTM_with_SoH/50%/fig/B48_LSTM_stacked.jpg')
plt.show()

# Print RMSE and MAE for the stacked model
stacked_rmse = math.sqrt(mean_squared_error(testY, stacked_predictions))
stacked_mae = mean_absolute_error(testY, stacked_predictions)
print('Stacked Model Test RMSE: %.3f' % stacked_rmse)
print('Stacked Model Test MAE: %.3f' % stacked_mae)
