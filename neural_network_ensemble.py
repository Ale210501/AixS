import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


def load(filepath):
    """Load data from CSV file."""
    data = pd.read_csv(filepath)
    return data


def removeColumns(data, cols):
    """Remove useless columns (zero variance, missing values)."""
    removedColumns = []
    data_cleaned = data.copy()
    
    print("\n=== ANALYZING COLUMNS FOR REMOVAL ===\n")
    
    for col in cols:
        if col == 'Label':
            continue
            
        if data[col].nunique() == 1:
            print(f"Removing '{col}': Zero variance")
            removedColumns.append(col)
            continue
        
        if data[col].std() == 0:
            print(f"Removing '{col}': Standard deviation is zero")
            removedColumns.append(col)
            continue
        
        missing_ratio = data[col].isnull().sum() / len(data)
        if missing_ratio > 0.5:
            print(f"Removing '{col}': Too many missing values ({missing_ratio*100:.2f}%)")
            removedColumns.append(col)
            continue
    
    if removedColumns:
        data_cleaned = data_cleaned.drop(columns=removedColumns)
        print(f"\nTotal columns removed: {len(removedColumns)}")
    else:
        print("\nNo columns to remove")
    
    return data_cleaned, removedColumns


def create_neural_network(input_dim, num_classes):
    """
    Create a fully connected neural network.
    Architecture based on common project requirements.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_neural_network(X_train, y_train, X_val, y_val, patience, learning_rate, epochs):
    """
    Train a neural network with given hyperparameters.
    Returns the trained model and training history.
    """
    # Create model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = create_neural_network(input_dim, num_classes)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history


def find_best_configuration(X_bootstrap, y_bootstrap, bootstrap_num):
    """
    Find the best neural network configuration using grid search.
    """
    print(f"\n{'='*80}")
    print(f"BOOTSTRAP {bootstrap_num}: FINDING BEST CONFIGURATION")
    print(f"{'='*80}")
    
    # Split bootstrap into train and validation (80-20)
    split_idx = int(0.8 * len(X_bootstrap))
    X_train = X_bootstrap[:split_idx]
    y_train = y_bootstrap[:split_idx]
    X_val = X_bootstrap[split_idx:]
    y_val = y_bootstrap[split_idx:]
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Hyperparameter grid
    patience_values = [5, 10, 15]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    epochs_values = [10, 20, 50, 100]
    
    best_config = {
        'patience': None,
        'learning_rate': None,
        'epochs': None,
        'val_loss': float('inf'),
        'model': None
    }
    
    config_num = 0
    total_configs = len(patience_values) * len(learning_rates) * len(epochs_values)
    
    # Grid search
    for patience in patience_values:
        for lr in learning_rates:
            for epochs in epochs_values:
                config_num += 1
                print(f"[{config_num}/{total_configs}] Testing: patience={patience}, "
                      f"lr={lr}, epochs={epochs}", end='')
                
                # Train model
                model, history = train_neural_network(
                    X_train, y_train, X_val, y_val,
                    patience, lr, epochs
                )
                
                # Get best validation loss
                val_loss = min(history.history['val_loss'])
                print(f" → val_loss={val_loss:.4f}")
                
                # Update best configuration
                if val_loss < best_config['val_loss']:
                    best_config['patience'] = patience
                    best_config['learning_rate'] = lr
                    best_config['epochs'] = epochs
                    best_config['val_loss'] = val_loss
                    best_config['model'] = model
                    print(f"  ✓ NEW BEST! val_loss={val_loss:.4f}")
    
    print(f"\nBest configuration for Bootstrap {bootstrap_num}:")
    print(f"  Patience: {best_config['patience']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Epochs: {best_config['epochs']}")
    print(f"  Val Loss: {best_config['val_loss']:.4f}")
    
    return best_config['model'], best_config


def majority_voting(predictions_list):
    """
    Combine predictions from multiple models using majority voting.
    """
    # Stack predictions: shape (n_models, n_samples)
    predictions_array = np.array(predictions_list)
    
    # Get majority vote for each sample
    # Use mode along axis 0 (across models)
    from scipy import stats
    majority_pred, _ = stats.mode(predictions_array, axis=0, keepdims=False)
    
    return majority_pred


# Main code
print("="*80)
print("NEURAL NETWORK ENSEMBLE WITH BAGGING")
print("="*80)

# Load training data
trainpath = "trainDdosLabelNumeric.csv"
print(f"\nLoading training data from: {trainpath}")
data = load(trainpath)
print(f"Training data shape: {data.shape}")

# Preprocessing - same as preprocessing.py
cols = list(data.columns.values)
data, removedColumns = removeColumns(data, cols)
print(f"\nShape after removing columns: {data.shape}")

# Separate features and labels
X = data.drop('Label', axis=1).values
y = data['Label'].values

print(f"\nFeature matrix: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("\n✓ Features normalized")

# Create 10 bootstraps with 80% of training set
print("\n" + "="*80)
print("CREATING BOOTSTRAPS")
print("="*80)

n_bootstraps = 10
bootstrap_size = int(0.8 * len(X))
bootstraps = []

for i in range(n_bootstraps):
    X_bootstrap, y_bootstrap = resample(X, y, n_samples=bootstrap_size, random_state=i)
    bootstraps.append((X_bootstrap, y_bootstrap))
    print(f"Bootstrap {i+1}: {len(X_bootstrap)} samples")

# Train neural networks on each bootstrap
print("\n" + "="*80)
print("TRAINING NEURAL NETWORKS")
print("="*80)

trained_models = []
best_configs = []

for i, (X_boot, y_boot) in enumerate(bootstraps, 1):
    best_model, best_config = find_best_configuration(X_boot, y_boot, i)
    trained_models.append(best_model)
    best_configs.append(best_config)

print("\n✓ All 10 neural networks trained successfully!")

# Load and preprocess test data
print("\n" + "="*80)
print("LOADING TEST DATA")
print("="*80)

testpath = "testDdosLabelNumeric.csv"
print(f"\nLoading test data from: {testpath}")
test_data = load(testpath)
print(f"Test data shape: {test_data.shape}")

# Remove same columns
test_data = test_data.drop(columns=removedColumns, errors='ignore')
print(f"Test data shape after column removal: {test_data.shape}")

# Separate features and labels
X_test = test_data.drop('Label', axis=1).values
y_test = test_data['Label'].values

# Normalize test data using same scaler
X_test = scaler.transform(X_test)
print(f"\nTest samples: {len(X_test)}")
print(f"Test class distribution:\n{np.unique(y_test, return_counts=True)}")

# Generate predictions from each model
print("\n" + "="*80)
print("GENERATING PREDICTIONS FROM ENSEMBLE")
print("="*80)

all_predictions = []
for i, model in enumerate(trained_models, 1):
    print(f"Model {i}: Generating predictions...", end='')
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    all_predictions.append(y_pred_classes)
    
    # Individual model accuracy
    acc = accuracy_score(y_test, y_pred_classes)
    print(f" Accuracy: {acc:.4f}")

# Combine predictions using majority voting
print("\n" + "="*80)
print("COMBINING PREDICTIONS WITH MAJORITY VOTING")
print("="*80)

ensemble_predictions = majority_voting(all_predictions)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print(f"\n✓ Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Confusion Matrix
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_test, ensemble_predictions)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(np.unique(y_test)),
            yticklabels=sorted(np.unique(y_test)))
plt.title('Confusion Matrix - Neural Network Ensemble', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Classification Report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print("\n" + classification_report(y_test, ensemble_predictions,
                                    target_names=[f"Class {i}" for i in sorted(np.unique(y_test))]))

# Summary
print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Number of models in ensemble: {len(trained_models)}")
print(f"Bootstrap size: {bootstrap_size} samples (80% of training)")
print(f"Test set size: {len(X_test)} samples")
print(f"\nIndividual model accuracies:")
for i, pred in enumerate(all_predictions, 1):
    acc = accuracy_score(y_test, pred)
    print(f"  Model {i}: {acc:.4f}")

print(f"\nEnsemble Accuracy (Majority Voting): {ensemble_accuracy:.4f}")
print(f"Average Individual Accuracy: {np.mean([accuracy_score(y_test, pred) for pred in all_predictions]):.4f}")
print(f"Improvement: {ensemble_accuracy - np.mean([accuracy_score(y_test, pred) for pred in all_predictions]):.4f}")
print("="*80)

print("\n✓ Neural Network Ensemble training and evaluation completed!")
