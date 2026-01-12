import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

train_path = "trainDdosLabelNumeric.csv"
test_path  = "testDdosLabelNumeric.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

X_train_full = train_df.drop("Label", axis=1).values
y_train_full = train_df["Label"].values

X_test = test_df.drop("Label", axis=1).values
y_test = test_df["Label"].values

scaler = MinMaxScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

num_classes = len(np.unique(y_train_full))
input_dim = X_train_full.shape[1]

def create_mlp(learning_rate):
    model = keras.Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,),
              kernel_initializer="glorot_uniform"),
        Dense(64, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(32, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def find_best_model(Xb, yb, bootstrap_id):

    X_tr, X_val, y_tr, y_val = train_test_split(
        Xb, yb, test_size=0.2, stratify=yb, random_state=SEED
    )

    patience_list = [5, 10, 15]
    lr_list = [0.0001, 0.001, 0.01, 0.1]
    epoch_list = [10, 20, 50, 100]

    best_val_loss = np.inf
    best_model = None
    best_params = None

    for p in patience_list:
        for lr in lr_list:
            for ep in epoch_list:

                model = create_mlp(lr)

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=p,
                    restore_best_weights=True,
                    verbose=0
                )

                history = model.fit(
                    X_tr, y_tr,
                    validation_data=(X_val, y_val),
                    epochs=ep,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )

                val_loss = min(history.history["val_loss"])

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = (p, lr, ep)

    return best_model

saved_models_dir = "saved_models"
os.makedirs(saved_models_dir, exist_ok=True)

n_bootstraps = 10
bootstrap_size = int(0.8 * len(X_train_full))
ensemble_models = []

for i in range(n_bootstraps):
    Xb, yb = resample(
        X_train_full,
        y_train_full,
        replace=True,
        n_samples=bootstrap_size,
        random_state=SEED + i
    )

    best_model = find_best_model(Xb, yb, i + 1)
    ensemble_models.append(best_model)

    model_path = os.path.join(saved_models_dir, f"mlp_bootstrap_{i+1}.h5")
    best_model.save(model_path)

all_predictions = []
for i, model in enumerate(ensemble_models, 1):
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    all_predictions.append(preds)

all_predictions = np.array(all_predictions)

ensemble_predictions = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=all_predictions
)

accuracy = accuracy_score(y_test, ensemble_predictions)

print(f"Accuracy Ensemble (Bagging MLP): {accuracy:.4f}")
print(confusion_matrix(y_test, ensemble_predictions))
print(classification_report(y_test, ensemble_predictions))