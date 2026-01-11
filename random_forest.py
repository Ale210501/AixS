import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns


def load(filepath):
    data = pd.read_csv(filepath)
    return data


def preElaborationData(data, cols):
    print("\n=== DATA PRE-ELABORATION ===\n")
    
    # Print description for all columns at once
    print("Statistical description of all variables:")
    print(data.describe())
    
    # Optional: print description for each column individually
    print("\n=== DETAILED DESCRIPTION FOR EACH VARIABLE ===\n")
    for col in cols:
        print(f"\n--- {col} ---")
        print(data[col].describe())
        print("-" * 50)


def removeColumns(data, cols):
    removedColumns = []
    data_cleaned = data.copy()
    
    print("\n=== ANALYZING COLUMNS FOR REMOVAL ===\n")
    
    for col in cols:
        # Skip the Label column (target variable)
        if col == 'Label':
            continue
            
        # Check for zero variance (all values are the same)
        if data[col].nunique() == 1:
            print(f"Removing '{col}': Zero variance (all values are the same)")
            removedColumns.append(col)
            continue
        
        # Check for columns with all zeros or very low variance
        if data[col].std() == 0:
            print(f"Removing '{col}': Standard deviation is zero")
            removedColumns.append(col)
            continue
        
        # Check for too many missing values (>50%)
        missing_ratio = data[col].isnull().sum() / len(data)
        if missing_ratio > 0.5:
            print(f"Removing '{col}': Too many missing values ({missing_ratio*100:.2f}%)")
            removedColumns.append(col)
            continue
    
    # Remove the identified columns
    if removedColumns:
        data_cleaned = data_cleaned.drop(columns=removedColumns)
        print(f"\nTotal columns removed: {len(removedColumns)}")
    else:
        print("\nNo columns to remove")
    
    return data_cleaned, removedColumns


def preElaborationClass(data, class_column):
    print(f"\n=== CLASS DISTRIBUTION ANALYSIS: {class_column} ===\n")
    
    # Count the occurrences of each class value
    class_counts = data[class_column].value_counts()
    print("Class value counts:")
    print(class_counts)
    
    # Calculate percentages
    class_percentages = data[class_column].value_counts(normalize=True) * 100
    print("\nClass value percentages:")
    print(class_percentages)
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    data[class_column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {class_column}', fontsize=16, fontweight='bold')
    plt.xlabel(class_column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTotal samples: {len(data)}")
    print(f"Number of classes: {data[class_column].nunique()}")



def stratifiedKfold(X, y, folds, seed):
    print(f"\n=== STRATIFIED K-FOLD CROSS VALIDATION ===")
    print(f"Number of folds: {folds}")
    print(f"Random seed: {seed}")
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1] if hasattr(X, 'shape') else 'N/A'}")
    print()
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    # Store all fold splits
    fold_splits = []
    
    # Generate splits
    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # Split data based on indices
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        else:
            X_train = X[train_index]
            X_test = X[test_index]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
        else:
            y_train = y[train_index]
            y_test = y[test_index]
        
        # Store the fold
        fold_splits.append((X_train, X_test, y_train, y_test))
        
        # Print fold information
        print(f"Fold {fold_num}:")
        print(f"  Training set size: {len(X_train)} samples")
        print(f"  Testing set size: {len(X_test)} samples")
        
        # Show class distribution in train and test sets
        if isinstance(y_train, pd.Series):
            train_dist = y_train.value_counts().sort_index()
            test_dist = y_test.value_counts().sort_index()
        else:
            train_dist = pd.Series(y_train).value_counts().sort_index()
            test_dist = pd.Series(y_test).value_counts().sort_index()
        
        print(f"  Train class distribution: {dict(train_dist)}")
        print(f"  Test class distribution: {dict(test_dist)}")
        print()
    
    print(f"Generated {len(fold_splits)} folds successfully")
    print("=" * 60)
    
    return fold_splits

def RFLearner(X, y, criterion='gini', max_features='sqrt', max_samples=1):
    clf = RandomForestClassifier(criterion=criterion,
                                 max_features=max_features,
                                 max_samples=max_samples,
                                 random_state=42)
    clf.fit(X, y)
    return clf


def RFLearner(X, y, criterion='gini', max_features='sqrt', max_samples=1):
    print(f"\n=== TRAINING RANDOM FOREST ===")
    print(f"Criterion: {criterion}")
    print(f"Max features: {max_features}")
    print(f"Max samples: {max_samples}")
    print(f"Training samples: {len(X)}")
    print(f"Number of features: {X.shape[1] if hasattr(X, 'shape') else 'N/A'}")
    
    # Create Random Forest classifier
    rf_model = RandomForestClassifier(
        criterion=criterion,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )
    
    # Train the model
    print("\nTraining Random Forest...")
    rf_model.fit(X, y)
    print("✓ Training completed!")
    
    # Print model information
    print(f"\nModel information:")
    print(f"  Number of trees: {rf_model.n_estimators}")
    print(f"  Number of classes: {rf_model.n_classes_}")
    print(f"  Feature importances available: {len(rf_model.feature_importances_)}")
    
    return rf_model


def evalF1(XTest, YTest, T):
    print(f"\n=== EVALUATING MODEL ===")
    print(f"Test samples: {len(XTest)}")
    
    # Make predictions
    y_pred = T.predict(XTest)
    
    # Calculate macro F1 score
    macro_f1 = f1_score(YTest, y_pred, average='macro')
    
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    
    # Additional evaluation metrics
    print(f"\nPer-class F1 scores:")
    class_f1_scores = f1_score(YTest, y_pred, average=None)
    unique_classes = sorted(set(YTest))
    for cls, score in zip(unique_classes, class_f1_scores):
        print(f"  Class {cls}: {score:.4f}")
    
    return macro_f1


def determineRFkFoldConfiguration(fold_splits):
    print("\n" + "=" * 80)
    print("DETERMINING BEST RANDOM FOREST CONFIGURATION")
    print("=" * 80)
    print("Testing configurations with 5-fold cross-validation...\n")
    
    # Define parameter grid
    criterions = ['gini', 'entropy']
    max_features_options = ['sqrt', 'log2', None]
    max_samples_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    best_config = {
        'criterion': None,
        'max_features': None,
        'max_samples': None,
        'avg_macro_f1': -1.0
    }
    
    all_results = []
    config_num = 0
    total_configs = len(criterions) * len(max_features_options) * len(max_samples_range)
    
    # Test all combinations
    for criterion in criterions:
        for max_features in max_features_options:
            for max_samples in max_samples_range:
                config_num += 1
                max_samples = round(max_samples, 1)
                
                print(f"[{config_num}/{total_configs}] Testing: criterion={criterion}, "
                      f"max_features={max_features}, max_samples={max_samples}")
                
                fold_f1_scores = []
                
                # Evaluate on each fold
                for fold_num, (X_train, X_test, y_train, y_test) in enumerate(fold_splits, 1):
                    # Train model with current configuration (suppress output)
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    model = RFLearner(X_train, y_train, criterion=criterion, 
                                    max_features=max_features, max_samples=max_samples)
                    f1 = evalF1(X_test, y_test, model)
                    
                    sys.stdout = old_stdout
                    
                    fold_f1_scores.append(f1)
                
                # Calculate average F1 across folds
                avg_f1 = np.mean(fold_f1_scores)
                std_f1 = np.std(fold_f1_scores)
                
                print(f"  → Average Macro F1: {avg_f1:.4f} (±{std_f1:.4f})")
                
                # Store results
                all_results.append({
                    'criterion': criterion,
                    'max_features': max_features,
                    'max_samples': max_samples,
                    'avg_macro_f1': avg_f1,
                    'std_macro_f1': std_f1
                })
                
                # Update best configuration if this is better
                if avg_f1 > best_config['avg_macro_f1']:
                    best_config['criterion'] = criterion
                    best_config['max_features'] = max_features
                    best_config['max_samples'] = max_samples
                    best_config['avg_macro_f1'] = avg_f1
                    print(f"  ✓ NEW BEST CONFIGURATION! Macro F1: {avg_f1:.4f}")
                print()
    
    # Print summary
    print("=" * 80)
    print("BEST CONFIGURATION FOUND")
    print("=" * 80)
    print(f"Criterion: {best_config['criterion']}")
    print(f"Max Features: {best_config['max_features']}")
    print(f"Max Samples: {best_config['max_samples']}")
    print(f"Average Macro F1: {best_config['avg_macro_f1']:.4f}")
    print("=" * 80)
    
    # Print top 5 configurations
    print("\nTop 5 Configurations:")
    sorted_results = sorted(all_results, key=lambda x: x['avg_macro_f1'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. criterion={result['criterion']}, max_features={result['max_features']}, "
              f"max_samples={result['max_samples']}: F1={result['avg_macro_f1']:.4f} "
              f"(±{result['std_macro_f1']:.4f})")
    print()
    
    return (best_config['criterion'], best_config['max_features'], 
            best_config['max_samples'], best_config['avg_macro_f1'])


# Main code
trainpath = "trainDdosLabelNumeric.csv"

# Load data
data = load(trainpath)
shape = data.shape
print(shape)
print(data.head())
print(data.columns)

# Pre-elaboration
cols = list(data.columns.values)
preElaborationData(data, cols)

# Remove useless columns
data, removedColumns = removeColumns(data, cols)
print("\n=== REMOVED COLUMNS ===")
print(removedColumns)
print(f"\nNew shape after removing columns: {data.shape}")

# Analyze class distribution
preElaborationClass(data, 'Label')


# Prepare data for K-fold cross validation
# Separate features (X) and target (y)
X = data.drop('Label', axis=1)
y = data['Label']

# Apply Stratified K-Fold Cross Validation
folds = 5
seed = 42
fold_splits = stratifiedKfold(X, y, folds, seed)

# Determine best Random Forest configuration
# best_criterion, best_max_features, best_max_samples, best_avg_f1 = determineRFkFoldConfiguration(fold_splits)
best_criterion="gini"
best_max_features="log2"
best_max_samples=0.5
best_avg_f1=0.9938
# Train final Random Forest model with best configuration on entire training set
print("\n" + "=" * 80)
print("TRAINING FINAL RANDOM FOREST WITH BEST CONFIGURATION")
print("=" * 80)
print(f"Using: criterion={best_criterion}, max_features={best_max_features}, max_samples={best_max_samples}")
print()

final_rf_model = RFLearner(X, y, criterion=best_criterion, 
                           max_features=best_max_features, 
                           max_samples=best_max_samples)

print("\n✓ Final model trained successfully!")
print(f"Expected Macro F1 Score (from CV): {best_avg_f1:.4f}")
print("=" * 80)

# Load and evaluate on test set
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

# Load test data
testpath = "testDdosLabelNumeric.csv"
print(f"\nLoading test data from: {testpath}")
test_data = load(testpath)
print(f"Test data shape: {test_data.shape}")

# Remove the same columns that were removed from training data
print(f"\nRemoving {len(removedColumns)} columns from test data...")
test_data = test_data.drop(columns=removedColumns, errors='ignore')
print(f"Test data shape after column removal: {test_data.shape}")

# Separate features and labels
X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

print(f"\nTest set: {len(X_test)} samples, {X_test.shape[1]} features")
print(f"Test class distribution:\n{y_test.value_counts().sort_index()}")

# Generate predictions
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)
y_pred = final_rf_model.predict(X_test)
print("✓ Predictions generated!")

# Calculate metrics
test_macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nTest Macro F1 Score: {test_macro_f1:.4f}")

# Confusion Matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.title('Confusion Matrix - Random Forest on Test Set', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Classification Report
print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print("\n" + classification_report(y_test, y_pred, 
                                    target_names=[f"Class {i}" for i in sorted(y_test.unique())]))

# Summary
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Training Macro F1 (CV): {best_avg_f1:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Difference: {abs(best_avg_f1 - test_macro_f1):.4f}")
print("=" * 80)