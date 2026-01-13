# AIxS - DDoS Attack Classification

Machine Learning project for DDoS attack detection and classification using multiple approaches.

## Overview

This project implements three different machine learning approaches to classify DDoS attacks:
- **Multi-Layer Perceptron (MLP)** - Deep learning approach with grid search
- **Random Forest Classifier** - Ensemble method with k-fold cross-validation
- **Bagging Neural Network** - Bootstrap aggregation with multiple MLPs

## Dataset

- Training set: `trainDdosLabelNumeric.csv` (10,000 samples)
- Test set: `testDdosLabelNumeric.csv` (1,000 samples)
- 5 classes of attacks (Labels: 0-4)
- 79 network traffic features

## Models

### 1. Multi-Layer Perceptron
- **Architecture**: 128 → 64 → 32 neurons (ReLU activation)
- **Optimization**: Grid search over learning rates, epochs, and batch sizes
- **Output**: Best model saved as `Best_MLP_Model.h5`

### 2. Random Forest Classifier
- **Method**: Stratified 5-fold cross-validation
- **Optimization**: Grid search over criterion, max_features, and max_samples
- **Results**: ~99.35% Macro F1 Score on test set

### 3. Bagging Neural Network
- **Approach**: 10 bootstrap samples with separate MLP models
- **Voting**: Majority voting for final predictions
- **Models**: Saved in `saved_models/` directory

## Requirements

```
pandas
numpy
tensorflow
keras
scikit-learn
matplotlib
seaborn
```

## Usage

```bash
# Run Random Forest Classifier
python Random_Forest_Classifier.py

# Run Multi-Layer Perceptron
python Multi-Layer_Perceptron.py

# Run Bagging Neural Network
python Bagging_Neural_Network.py
```

## Results

All models achieve high performance (>99% accuracy) on the DDoS classification task, with detailed confusion matrices and classification reports generated for each approach.


