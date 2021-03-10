# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:53:34 2021

@author: Amine
"""
import math
import numpy as np # linear algebra
#repésentation des résultats
import matplotlib.pyplot as plt
#Machine learning
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN, tcn_full_summary
import array_from_uat as uat


#DATA
WEATHER_STA = 14578001 #numero de la station

#periode 
START = "2016-01-01"
STOP = "2016-01-12"

#la fenetre de la prevision
HISTORY_LAG = 10
lag=HISTORY_LAG

#predict serie or delta
DELTA = False

# les parametre pour le model
verbose=1
epochs=200    #époques
batch_size=16

#importe data
serie = uat.get_array(uat.TEMPERATURE, WEATHER_STA, START, STOP)
serie = serie.transpose()
if DELTA:
    features = serie[1:] - serie[:-1]
else:
    features = serie
if verbose > 0:
    print(features.shape)

#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

#Pour le training on prend 75% des données initials
training_dataset_length = math.ceil(len(features) * .75)
train_data = scaled_data[0:training_dataset_length, : ]

#Splitting the data x_train et y_train
x_train = []
y_train = []
for i in range(lag, len(train_data)):
    x_train.append(train_data[i-lag:i])
    y_train.append(train_data[i])
x_train = np.array(x_train)
y_train = np.array(y_train)
if verbose > 0:
    print(x_train.shape)
    print(y_train.shape)



#Test data set
test_data = scaled_data

#splitting the x_test and y_test data sets
x_test = []
if DELTA:
    y_test = serie[lag+1 :, : ]
    previous = serie[lag: -1, : ]
else:
    y_test = serie[lag :, : ]
    previous = serie[lag-1: -1, : ]
for i in range(lag, len(test_data)):
    x_test.append(test_data[i-lag:i])
    
x_test = np.array(x_test)

#TCN

i = Input(shape=(lag , 1))
m = TCN()(i)
m = TCN(nb_filters=8)(i)

m = Dense(1, activation='linear')(m)


model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile('adam', 'mae')

print('Train...')
model.fit( x_train , y_train, epochs=100, verbose=2)

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

#check predicted values
predict = model.predict(x_test)
#keep only next predicted value
predict = predict[:,-1]
predict = predict.reshape(-1, 1)
#Undo scaling

predict = scaler.inverse_transform(predict)
#add back previous point
if DELTA:
    predict += serie[lag:-1, :]


#RMSE et affichage
def rmse(predict, ref, training_length, lag):
    """Compute Root Mean Square Error after training data"""
    return np.sqrt(np.mean((predict[training_length+lag:] - ref[training_length+lag:])**2))




result = rmse(predict, y_test, training_dataset_length, lag)
if verbose == 1:
    print("use previous RMSE:",
          rmse(previous, y_test, training_dataset_length, lag))
    print("predicted RMSE:", result)
    print("previous vs predicted RMSE:",
          rmse(predict, previous, training_dataset_length, lag))
  
plt.figure(1)
plt.plot(predict, color='red', label='predicted', linewidth=1.0)
plt.plot(y_test, color='blue', label='actual', linewidth=1.0)
# plt.axvline(x=training_dataset_length/10-lag)
plt.legend(['predicted', 'actual'])
plt.show()

y_test1=y_test[0:int(len(test_data)/10)]
predict1=predict[0:int(len(test_data)/10)]

plt.figure(2)
plt.plot(predict1, color='red', label='predicted', linewidth=1.0)
plt.plot(y_test1, color='blue', label='actual', linewidth=1.0)
plt.legend(['predicted', 'actual'])
plt.show()

y_test1=y_test[0:int(len(test_data)/240)]
predict1=predict[0:int(len(test_data)/240)]
plt.figure(3)
plt.plot(predict1, color='red', label='predicted', linewidth=1.0)
plt.plot(y_test1, color='blue', label='actual', linewidth=1.0)
plt.legend(['predicted', 'actual'])
plt.show()



