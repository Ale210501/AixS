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
    print("Statistical description of all variables:")
    print(data.describe())
    
    for col in cols:
        print(f"\n{col} description:")
        print(data[col].describe())


def removeColumns(data, cols):
    removedColumns = []
    data_cleaned = data.copy()
    
    for col in cols:
        if col == 'Label':
            continue
            
        if data[col].nunique() == 1:
            print(f"Removing '{col}': Zero variance (all values are the same)")
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


def preElaborationClass(data, class_column):
    print(f"\n Class distribution analysis: '{class_column}'")
    
    class_counts = data[class_column].value_counts()
    print("Class value counts:")
    print(class_counts)
    
    class_percentages = data[class_column].value_counts(normalize=True) * 100
    print("\nClass value percentages:")
    print(class_percentages)
    
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
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    
    fold_splits = []
    
    for fold_num, (train_index, test_index) in enumerate(skf.split(X, y), 1):
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
        
        fold_splits.append((X_train, X_test, y_train, y_test))
        
        if isinstance(y_train, pd.Series):
            train_dist = y_train.value_counts().sort_index()
            test_dist = y_test.value_counts().sort_index()
        else:
            train_dist = pd.Series(y_train).value_counts().sort_index()
            test_dist = pd.Series(y_test).value_counts().sort_index()
    
    print(f"Generated {len(fold_splits)} folds successfully")
    
    return fold_splits

def RFLearner(X, y, criterion='gini', max_features='sqrt', max_samples=1):
    clf = RandomForestClassifier(criterion=criterion,
                                 max_features=max_features,
                                 max_samples=max_samples,
                                 random_state=42)
    clf.fit(X, y)
    return clf


def RFLearner(X, y, criterion='gini', max_features='sqrt', max_samples=1):    
    rf_model = RandomForestClassifier(
        criterion=criterion,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )
    
    rf_model.fit(X, y)    
    return rf_model


def evalF1(XTest, YTest, T):    
    y_pred = T.predict(XTest)
    macro_f1 = f1_score(YTest, y_pred, average='macro')
    return macro_f1


def determineRFkFoldConfiguration(fold_splits):
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
    
    for criterion in criterions:
        for max_features in max_features_options:
            for max_samples in max_samples_range:
                config_num += 1
                max_samples = round(max_samples, 1)
                
                print(f"Testing: criterion={criterion}, max_features={max_features}, max_samples={max_samples}")
                
                fold_f1_scores = []
                
                for fold_num, (X_train, X_test, y_train, y_test) in enumerate(fold_splits, 1):
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    model = RFLearner(X_train, y_train, criterion=criterion, 
                                    max_features=max_features, max_samples=max_samples)
                    f1 = evalF1(X_test, y_test, model)
                    
                    sys.stdout = old_stdout
                    
                    fold_f1_scores.append(f1)
                
                avg_f1 = np.mean(fold_f1_scores)
                std_f1 = np.std(fold_f1_scores)
                
                print(f"Average Macro F1: {avg_f1:.4f}")
                
                all_results.append({
                    'criterion': criterion,
                    'max_features': max_features,
                    'max_samples': max_samples,
                    'avg_macro_f1': avg_f1,
                    'std_macro_f1': std_f1
                })
                
                if avg_f1 > best_config['avg_macro_f1']:
                    best_config['criterion'] = criterion
                    best_config['max_features'] = max_features
                    best_config['max_samples'] = max_samples
                    best_config['avg_macro_f1'] = avg_f1
                    print(f"New best configuration!")
                print()
    
    print("Best configuration found:")
    print(f"Criterion: {best_config['criterion']}")
    print(f"Max Features: {best_config['max_features']}")
    print(f"Max Samples: {best_config['max_samples']}")
    print(f"Average Macro F1: {best_config['avg_macro_f1']:.4f}")
    
    return (best_config['criterion'], best_config['max_features'], 
            best_config['max_samples'], best_config['avg_macro_f1'])

trainpath = "trainDdosLabelNumeric.csv"

data = load(trainpath)
shape = data.shape
print(shape)
print(data.head())
print(data.columns)

cols = list(data.columns.values)
preElaborationData(data, cols)

data, removedColumns = removeColumns(data, cols)
print(f"\nNew shape after removing columns: {data.shape}")

preElaborationClass(data, 'Label')

X = data.drop('Label', axis=1)
y = data['Label']

folds = 5
seed = 42
fold_splits = stratifiedKfold(X, y, folds, seed)

best_criterion, best_max_features, best_max_samples, best_avg_f1 = determineRFkFoldConfiguration(fold_splits)

final_rf_model = RFLearner(X, y, criterion=best_criterion, 
                           max_features=best_max_features, 
                           max_samples=best_max_samples)

testpath = "testDdosLabelNumeric.csv"
test_data = load(testpath)

test_data = test_data.drop(columns=removedColumns, errors='ignore')

X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

y_pred = final_rf_model.predict(X_test)

test_macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nTest Macro F1 Score: {test_macro_f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.title('Confusion Matrix - Random Forest on Test Set', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print("Classification report:")
print("\n" + classification_report(y_test, y_pred, 
                                    target_names=[f"Class {i}" for i in sorted(y_test.unique())]))