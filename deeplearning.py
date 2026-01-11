from google.colab import drive
drive.mount('/content/drive')

# function to split the dataset in train X and train Y
def getXY(dataset, label):
  train_Y=dataset[label]
  print(train_Y.head)

  #remove label from dataset
  train_X=dataset.drop(label, axis=1)
  train_X=train_X.values

  return train_X, train_Y

def printPlotLoss(history, d):
  loss=history.history['loss']
  val_loss=history.history['val_loss']
  epochs=range(1, len(loss)+1)
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig("plotLoss"+str(d)+".png")
  plt.show()
  plt.close()

def printPlotAccuracy(history, d):
  acc=history.history['accuracy']
  val_acc=history.history['val_accuracy']
  epochs=range(1, len(acc)+1)
  plt.plot(epochs, acc, 'b', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.savefig("plotAccuracy"+str(d)+".png")
  plt.show()
  plt.close()

import pandas as pd
import numpy as np

pathTrain='/content/drive/MyDrive/dataset/trainDdosLabelNumeric.csv'
pathTest='/content/drive/MyDrive/dataset/testDdosLabelNumeric.csv'
train=pd.read_csv(pathTrain)
test=pd.read_csv(pathTest)

print(train.head(8))

my_seed=42
import tensorflow as tf
tf.random.set_seed(my_seed)

train_X, train_Y=getXY(train, 'Label')
test_X, test_Y=getXY(test, 'Label')
print(train_X.shape)
print(np.unique(train_Y))
#print(type(train_X))
#print(type(train))
#print(train_X)

print("Max value before preprocessing:"+str(np.amax(train_X)))
print("Max value before preprocessing test:"+str(np.amax(test_X)))
print("Min value before prepocessing:"+str(np.amin(train_X)))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.fit_transform(test_X)
print("Max value after preprocessing:"+str(np.amax(train_X)))
print("Min value after preprocessing:"+str(np.amin(train_X)))
print("Shape of")

from keras.src.models import model
from tensorflow import keras
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras import callbacks

def MLP_architecture(train_X):
  n_col=train_X.shape[1]
  input=Input(shape=(n_col,))

  hidden1=Dense(128, activation='relu', kernel_initializer='glorot_uniform', name='l1')(input)
  hidden2=Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='l2')(hidden1)
  hidden3=Dense(32, activation='relu', kernel_initializer='glorot_uniform', name='l3')(hidden2)
  output=Dense(5, activation='softmax')(hidden3)
  model=Model(inputs=input, outputs=output)
  model.summary()
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
  return model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

model=MLP_architecture(train_X)

callbacks_list=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True), ]

XTraining, XValidation, YTraining, YValidation=train_test_split(train_X, train_Y, stratify=train_Y, test_size=0.2)
print(YTraining[0:3])
print(YValidation.shape)
YTraining=tf.keras.utils.to_categorical(YTraining, 5)
YValidation=tf.keras.utils.to_categorical(YValidation, 5)
print(YTraining[0:3])
print(YValidation.shape)

history=model.fit(XTraining, YTraining, batch_size=32, epochs=150, verbose=2, callbacks=callbacks_list, shuffle=True, validation_data=(XValidation, YValidation))

printPlotAccuracy(history, 'MLP')
printPlotLoss(history, 'MLP')

from sklearn.metrics import classification_report, confusion_matrix
y_predicted=model.predict(test_X, verbose=0)
print('Print of the probability of sample n.1')
print(y_predicted[0])
y_predicted=np.argmax(y_predicted, axis=1)
print('Print of the predicted class of sample n.1')
print(y_predicted[0])

cm=confusion_matrix(test_Y, y_predicted)
print('Confusion matrix: ')
print(cm)
print('Classification test: ')
print(classification_report(test_Y, y_predicted))

y_predicted=model.predict(train_X, verbose=0)
y_predicted=np.argmax(y_predicted, axis=1)
cm = confusion_matrix(train_Y, y_predicted)
print('Confusion matrix: ')
print(cm)
print('Classification train: ')
print(classification_report(train_Y, y_predicted))