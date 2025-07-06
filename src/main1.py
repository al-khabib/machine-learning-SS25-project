from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from random_person_generator import random_person

# data loading

df = pd.read_csv('bodyPerformance.csv')

# EDA


def run_eda(data):
    print("First 5 rows:")
    print(data.head())

    print("\nData Info:")
    print(data.info())

    print("\nMissing values per column:")
    print(data.isnull().sum())

    print("\nClass distribution:")
    print(data['class'].value_counts())

    # Visualizations
    plt.figure(figsize=(8, 4))
    sns.countplot(x='class', data=data, palette='Set2')
    plt.title('Class Distribution')
    plt.show()

    # Gender distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(x='gender', data=data, hue='class', palette='Set1')
    plt.title('Gender Distribution by Class')
    plt.show()

    # Correlation heatmap (numerical features)
    plt.figure(figsize=(10, 8))
    num_cols = data.select_dtypes(include=[np.number]).columns
    corr = data[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot for selected features
    # sample_cols = ['age', 'height_cm', 'weight_kg',
    #                'body fat_%', 'gripForce', 'sit-ups counts', 'class']
    # sns.pairplot(data[sample_cols], hue='class', corner=True)
    # plt.suptitle('Pairplot of Selected Features', y=1.02)
    # plt.show()


# Data preprocessing
# Encode categorical variables
le_gender = LabelEncoder()
le_class = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['class'] = le_class.fit_transform(df['class'])  # For y

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101, stratify=y
)

# feature normalization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Engineering

# Add BMI as a new feature (weight_kg / (height_m)^2)


def add_bmi(X, col_height='height_cm', col_weight='weight_kg'):
    X = X.copy()
    X['BMI'] = X[col_weight] / (X[col_height] / 100) ** 2
    return X


X_train_fe = add_bmi(pd.DataFrame(X_train, columns=X.columns))
X_test_fe = add_bmi(pd.DataFrame(X_test, columns=X.columns))

# Re-normalize after feature engineering
X_train_fe_scaled = scaler.fit_transform(X_train_fe)
X_test_fe_scaled = scaler.transform(X_test_fe)


#
# Model Training and Evaluation
def train_and_evaluate_models(X_tr, X_te, y_tr, y_te, class_labels):
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr')
    }
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
        print(f"\n{name} Accuracy: {acc:.4f}")
    return results


results = train_and_evaluate_models(
    X_train_fe_scaled, X_test_fe_scaled, y_train, y_test, le_class.classes_)

# Confusion Matrices and Comparison


def plot_confusion_matrices(results, class_labels):
    n = len(results)
    plt.figure(figsize=(6 * n, 5))
    for idx, (name, res) in enumerate(results.items()):
        plt.subplot(1, n, idx + 1)
        disp = ConfusionMatrixDisplay(
            res['confusion_matrix'], display_labels=class_labels)
        disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
        plt.title(f"{name}\nAccuracy: {res['accuracy']:.2f}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=accuracies, palette='Set2')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()


#  Summary Table
def print_summary_table(results):
    print("\nModel Performance Summary:")
    print(f"{'Model':<20} {'Accuracy':>10}")
    print("-" * 32)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:.4f}")


# hyperparameter tuning


def tune_knn(X_tr, y_tr):
    param_grid = {'n_neighbors': list(range(1, 22, 2))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=5, scoring='accuracy')
    grid.fit(X_tr, y_tr)
    print("Best k:", grid.best_params_['n_neighbors'])
    print("Best cross-validated accuracy:", grid.best_score_)
    return grid.best_estimator_


def tune_rf():
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the classifier
    rf = RandomForestClassifier()

    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )

    # Fitting
    grid_search.fit(X_train_fe_scaled, y_train)

    # Output the best parameters and score
    print("Best hyperparameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")


def predict_personal_performance(model, scaler, le_gender, le_class, input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input['gender'] = le_gender.transform(df_input['gender'])
    df_input = add_bmi(df_input)
    X_input_scaled = scaler.transform(df_input)
    pred = model.predict(X_input_scaled)
    pred_label = le_class.inverse_transform(pred)
    print(f"Predicted body performance class: {pred_label[0]}")
    return pred_label[0]


if __name__ == "__main__":
    run_eda(df)
    plot_confusion_matrices(results, le_class.classes_)
    plot_model_comparison(results)
    print_summary_table(results)
    tune_knn(X_train_fe_scaled, y_train)
    tune_rf()
    predict_personal_performance()

# hyperparameter optimazation
# feature engineering
# feature selection
# knn support vector machine with HP optimazation
