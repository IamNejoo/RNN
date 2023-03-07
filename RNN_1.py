import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, SimpleRNN, GRU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import math

training_data = [
                    {'name': '/home/nejoo/prueba_red_neuronal/data/gsk80.csv', 'train':1, 'test':0},
                    {'name': '/home/nejoo/prueba_red_neuronal/data/gsk85.csv', 'train':0, 'test':1},
                    {'name': '/home/nejoo/prueba_red_neuronal/data/gsk90.csv', 'train':1, 'test':0},
                    {'name': '/home/nejoo/prueba_red_neuronal/data/gsk95.csv', 'train':0, 'test':1},
                    {'name': '/home/nejoo/prueba_red_neuronal/data/gsk100.csv', 'train':1, 'test':0}
]


x_train = []
y_train = []

x_test = []
y_test = []

for data in training_data:
    with open(data['name'],'r') as datos:
        reader = csv.reader(datos,delimiter=';')

        for row in reader:
            if data['train'] == 1:
                x_train.append(row[1:len(row)])
                y_train.append(row[0])

            if data['test'] == 1:
                x_test.append(row[1:len(row)])
                y_test.append(row[0])

    for i in range(0,len(y_train)):
        x_train[i] = [float(j) for j in x_train[i]]
        y_train[i] = float(y_train[i])

    for i in range(0,len(y_test)):
        x_test[i] = [float(j) for j in x_test[i]]
        y_test[i] = float(y_test[i])

#normaliza los datos x_train y y_train, x_test y y_test

#scaler = MinMaxScaler(feature_range=(-1, 1))
#x_train = scaler.fit_transform(x_train)
#y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
#x_test = scaler.fit_transform(x_test)
#y_test = scaler.fit_transform(np.array(y_test).reshape(-1,1))


def lstm_model(x_train, y_train, x_test, y_test):
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    dim_salida = 1
    na = 50
    model = Sequential()
    model.add(LSTM(na, input_shape=(1, x_train.shape[2]), return_sequences=True))
    model.add(Dense(dim_salida, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=4000, batch_size=100, validation_data=(x_test, y_test))
    # mostrar historial de entrenamiento
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return 0 

x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)

lstm_model(x_train, y_train, x_test, y_test)

