import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def getXY(dataset, label):
    train_Y = dataset[label]
    train_X = dataset.drop(label, axis=1).values
    return train_X, train_Y

def printPlotLoss(history, d):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f"plotLoss_{d}.png")
    plt.show()
    plt.close()

def printPlotAccuracy(history, d):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f"plotAccuracy_{d}.png")
    plt.show()
    plt.close()

def MLP_architecture(train_X):
    n_col = train_X.shape[1]
    input_layer = Input(shape=(n_col,))
    hidden1 = Dense(128, activation='relu', kernel_initializer='glorot_uniform', name='l1')(input_layer)
    hidden2 = Dense(64, activation='relu', kernel_initializer='glorot_uniform', name='l2')(hidden1)
    hidden3 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', name='l3')(hidden2)
    output = Dense(5, activation='softmax')(hidden3)
    model = Model(inputs=input_layer, outputs=output)
    return model

pathTrain = 'trainDdosLabelNumeric.csv'
pathTest  = 'testDdosLabelNumeric.csv'

train = pd.read_csv(pathTrain)
test  = pd.read_csv(pathTest)

my_seed = 42
tf.random.set_seed(my_seed)

train_X, train_Y = getXY(train, 'Label')
test_X, test_Y   = getXY(test, 'Label')

scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)
test_X  = scaler.transform(test_X)

learning_rates = [0.0001, 0.001, 0.01, 0.1]
epochs_list    = [30, 60, 90, 120, 150]
batch_sizes    = [16, 32, 64, 128]

best_val_loss = np.inf
best_params = {}

XTraining, XValidation, YTraining, YValidation = train_test_split(
    train_X, train_Y, stratify=train_Y, test_size=0.2, random_state=my_seed
)
YTraining = tf.keras.utils.to_categorical(YTraining, 5)
YValidation = tf.keras.utils.to_categorical(YValidation, 5)

for lr, ep, bs in product(learning_rates, epochs_list, batch_sizes):
    model = MLP_architecture(train_X)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    history = model.fit(
        XTraining, YTraining,
        batch_size=bs,
        epochs=ep,
        verbose=0,
        validation_data=(XValidation, YValidation),
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    
    val_loss = min(history.history['val_loss'])
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {'learning_rate': lr, 'epochs': ep, 'batch_size': bs}
        best_model = model
        best_history = history

print(f"\nBest Parameters: {best_params}")
print(f"Best Validation Loss: {best_val_loss:.4f}")

history = best_history
printPlotAccuracy(history, 'BestMLP')
printPlotLoss(history, 'BestMLP')

y_pred_test = best_model.predict(test_X, verbose=0)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

cm_test = confusion_matrix(test_Y, y_pred_test_classes)
print('Confusion matrix (Test):\n', cm_test)
print('Classification Report (Test):\n', classification_report(test_Y, y_pred_test_classes))

y_pred_train = best_model.predict(train_X, verbose=0)
y_pred_train_classes = np.argmax(y_pred_train, axis=1)
cm_train = confusion_matrix(train_Y, y_pred_train_classes)
print('Confusion matrix (Train):\n', cm_train)
print('Classification Report (Train):\n', classification_report(train_Y, y_pred_train_classes))

model_save_path = './Best_MLP_Model.h5'
best_model.save(model_save_path)