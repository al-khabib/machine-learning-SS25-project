from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import cdist
from collections import Counter
import numpy as np
from preprocessing import BodyPerformancePreprocessor


processor = BodyPerformancePreprocessor()
df_clean = processor.load_and_clean_data("bodyPerformance.csv")

# Engineer features
df_engineered = processor.engineer_features(df_clean)

# Prepare features and target
X, y = processor.prepare_features_target(df_engineered)

# Split and scale data
X_train, X_test, y_train, y_test = processor.split_and_scale_data(X, y)

# Feature selection
selected_indices = processor.forward_selection(
    X_train, y_train, max_features=12)

# Get final feature sets
X_train_final = processor.get_selected_features(
    X_train, selected_indices)
X_test_final = processor.get_selected_features(X_test, selected_indices)

X_train = X_train_final
X_test = X_test_final


def engineer_features(self, df):
    df_eng = df.copy()

    # Encode categorical variables
    df_eng['gender_numeric'] = df_eng['gender'].map({'M': 1, 'F': 0})

    # Domain-specific features
    df_eng['BMI'] = df_eng['weight_kg'] / ((df_eng['height_cm']/100) ** 2)
    df_eng['strength_to_weight'] = df_eng['gripForce'] / \
        df_eng['weight_kg']
    df_eng['flexibility_ratio'] = df_eng['sit and bend forward_cm'] / \
        df_eng['height_cm']
    df_eng['endurance_strength_ratio'] = df_eng['sit-ups counts'] / \
        df_eng['gripForce']
    df_eng['power_to_weight'] = df_eng['broad jump_cm'] / \
        df_eng['weight_kg']
    df_eng['age_group'] = pd.cut(df_eng['age'], bins=[
        0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]).astype(float)

    # Polynomial features
    key_features = ['BMI', 'strength_to_weight',
                    'flexibility_ratio', 'age']
    for feature in key_features:
        df_eng[f"{feature}_squared"] = df_eng[feature] ** 2

    # Interaction terms
    interactions = [
        ('strength_to_weight', 'power_to_weight', 'strength_power_interaction'),
        ('BMI', 'age', 'BMI_age_interaction'),
        ('sit-ups counts', 'broad jump_cm', 'endurance_power_interaction'),
        ('body fat_%', 'gripForce', 'bodyfat_strength_interaction'),
        ('age', 'flexibility_ratio', 'age_flexibility_interaction')
    ]

    for feat1, feat2, interaction_name in interactions:
        df_eng[interaction_name] = df_eng[feat1] * df_eng[feat2]

    print(
        f"Feature engineering completed: {df_eng.shape[1]} total features")
    return df_eng


# Get feature names and class mapping
selected_feature_names = [processor.feature_names[i]
                          for i in selected_indices]
class_mapping = processor.get_class_mapping()


def knn_predict(X_train, y_train, X_test, k=5, metric='euclidean'):

    predictions = []

    # Calculate distances from each test point to all training points
    distances = cdist(X_test, X_train, metric=metric)

    for i in range(len(X_test)):
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]

        # Get labels of k nearest neighbors
        nearest_labels = y_train[nearest_indices]

        # Predict using majority vote
        prediction = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(prediction)

    return np.array(predictions)


def test_knn_values(X_train, y_train, X_val, y_val, k_values=[1, 3, 5, 7, 9, 11, 15, 19]):
    results = {}

    for k in k_values:
        predictions = knn_predict(X_train, y_train, X_val, k=k)
        accuracy = accuracy_score(y_val, predictions)
        results[k] = accuracy
        print(f"k={k}: Validation Accuracy = {accuracy:.4f}")

    # Find best k
    best_k = max(results, key=results.get)
    print(f"\nBest k: {best_k} with accuracy: {results[best_k]:.4f}")

    return best_k, results
