import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class BodyPerformancePreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.feature_names = None

    def load_and_clean_data(self, filepath):

        # Load data
        df = pd.read_csv(filepath)

        # Data quality assessment
        print("Data Quality Assessment:")
        print(f"Original shape: {df.shape}")

        # Treat zero values in physiological measurements as missing
        zero_cols = ['diastolic', 'systolic', 'gripForce', 'broad jump_cm']
        for col in zero_cols:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                df[col] = df[col].replace(0, np.nan)
                print(f"- {col}: {zero_count} zeros treated as missing")

        # Gender-specific median imputation
        for col in zero_cols:
            if df[col].isnull().sum() > 0:
                male_median = df[df['gender'] == 'M'][col].median()
                female_median = df[df['gender'] == 'F'][col].median()

                mask_male = (df['gender'] == 'M') & (df[col].isnull())
                mask_female = (df['gender'] == 'F') & (df[col].isnull())

                df.loc[mask_male, col] = male_median
                df.loc[mask_female, col] = female_median

        print(f"Missing values after imputation: {df.isnull().sum().sum()}")
        return df

    def engineer_features(self, df):
        df_eng = df.copy()

        # Encode categorical variables
        df_eng['gender_numeric'] = df_eng['gender'].map({'M': 1, 'F': 0})

        # Domain-specific features
        df_eng['BMI'] = df_eng['weight_kg'] / ((df_eng['height_cm']/100) ** 2)
        df_eng['strength_to_weight'] = df_eng['gripForce'] / df_eng['weight_kg']
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

    def prepare_features_target(self, df):
        """
        Prepare feature matrix and target variable
        """
        # Feature columns (exclude original categorical columns)
        feature_columns = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic',
                           'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm',
                           'gender_numeric', 'BMI', 'strength_to_weight', 'flexibility_ratio',
                           'endurance_strength_ratio', 'power_to_weight', 'age_group',
                           'BMI_squared', 'strength_to_weight_squared', 'flexibility_ratio_squared', 'age_squared',
                           'strength_power_interaction', 'BMI_age_interaction', 'endurance_power_interaction',
                           'bodyfat_strength_interaction', 'age_flexibility_interaction']

        X = df[feature_columns]
        y = self.label_encoder.fit_transform(df['class'])

        self.feature_names = list(X.columns)
        return X.values, y

    def split_and_scale_data(self, X, y, test_size=0.2, random_state=101):

        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit scaler ONLY on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(
            f"Data split: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        print("Feature normalization: Parameters computed only on training data")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def forward_selection(self, X_train, y_train, max_features=15):
        """
        Implement forward selection algorithm
        """
        print("Running Forward Selection ")

        selected_features = []
        remaining_features = list(range(X_train.shape[1]))
        best_score = 0

        base_classifier = LogisticRegression(random_state=42, max_iter=1000)

        for iteration in range(min(max_features, X_train.shape[1])):
            best_feature = None
            best_iteration_score = 0

            for feature_idx in remaining_features:
                temp_features = selected_features + [feature_idx]
                X_temp = X_train[:, temp_features]

                cv_scores = cross_val_score(
                    base_classifier, X_temp, y_train, cv=5, scoring='accuracy')
                current_score = cv_scores.mean()

                if current_score > best_iteration_score:
                    best_iteration_score = current_score
                    best_feature = feature_idx

            if best_iteration_score > best_score:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_score = best_iteration_score
                print(
                    f"  Iteration {iteration + 1}: Added {self.feature_names[best_feature]} (Score: {best_iteration_score:.4f})")
            else:
                print(f"  Iteration {iteration + 1}: No improvement, stopping")
                break

        self.selected_features = selected_features
        return selected_features

    def get_selected_features(self, X, feature_indices=None):
        """
        Get features based on selection indices
        """
        if feature_indices is None:
            feature_indices = self.selected_features

        return X[:, feature_indices]

    def get_class_mapping(self):
        """
        Get class label mapping
        """
        return dict(zip(self.label_encoder.classes_,
                        self.label_encoder.transform(self.label_encoder.classes_)))

# Example usage


# def preprocess_body_performance_data(filepath):
    # """
    # Complete preprocessing pipeline for body performance data
    # """
    # preprocessor = BodyPerformancePreprocessor()

    # # Load and clean data
    # df_clean = preprocessor.load_and_clean_data(filepath)

    # # Engineer features
    # df_engineered = preprocessor.engineer_features(df_clean)

    # # Prepare features and target
    # X, y = preprocessor.prepare_features_target(df_engineered)

    # # Split and scale data
    # X_train, X_test, y_train, y_test = preprocessor.split_and_scale_data(X, y)

    # # Feature selection
    # selected_indices = preprocessor.forward_selection(
    #     X_train, y_train, max_features=12)

    # # Get final feature sets
    # X_train_final = preprocessor.get_selected_features(
    #     X_train, selected_indices)
    # X_test_final = preprocessor.get_selected_features(X_test, selected_indices)

    # # Get feature names and class mapping
    # selected_feature_names = [preprocessor.feature_names[i]
    #                           for i in selected_indices]
    # class_mapping = preprocessor.get_class_mapping()

    # return {
    #     'X_train': X_train_final,
    #     'X_test': X_test_final,
    #     'y_train': y_train,
    #     'y_test': y_test,
    #     'feature_names': selected_feature_names,
    #     'class_mapping': class_mapping,
    #     'preprocessor': preprocessor
    # }


# Recommended feature set from forward selection analysis
RECOMMENDED_FEATURES = [
    'sit and bend forward_cm',
    'sit-ups counts',
    'age',
    'weight_kg',
    'gripForce',
    'gender_numeric',
    'flexibility_ratio_squared',
    'BMI_squared',
    'body fat_%',
    'power_to_weight',
    'age_flexibility_interaction',
    'strength_to_weight_squared'
]


# if __name__ == "__main__":
#     # Example usage
#     filepath = 'bodyPerformance.csv'
#     preprocessed_data = preprocess_body_performance_data(filepath)

#     print("Preprocessing complete.")
#     print(f"Selected features: {preprocessed_data['feature_names']}")
#     print(f"Class mapping: {preprocessed_data['class_mapping']}")
#     print(f"Train set shape: {preprocessed_data['X_train'].shape}")
#     print(f"Test set shape: {preprocessed_data['X_test'].shape}")
