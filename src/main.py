"""
Body Performance Dataset Preprocessing Module

This module provides functions for preprocessing the Body Performance dataset
for multiclass classification tasks. It handles:
- Missing/zero value imputation
- Feature engineering (BMI, strength-to-weight, age groups)
- Categorical variable encoding
- Feature normalization/standardization
- Train/validation/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def preprocess_body_performance_data(df, random_state=42, test_size=0.2, val_size=0.2):
    """
    Preprocess the body performance dataset for machine learning.

    Parameters:
    -----------
    df : pandas.DataFrame
        The raw body performance dataset
    random_state : int, default=42
        Random state for reproducibility
    test_size : float, default=0.2
        Proportion of data to use for testing
    val_size : float, default=0.2
        Proportion of non-test data to use for validation

    Returns:
    --------
    dict
        Dictionary containing preprocessed data splits and metadata
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()

    # 1. Handle missing/zero values in biological measurements
    # For the few zero values in physiological measurements that should never be zero
    zero_cols = ['diastolic', 'systolic', 'gripForce', 'broad jump_cm']
    for col in zero_cols:
        # Replace zeros with NaN then impute using median of the same gender
        data[col] = data[col].replace(0, np.nan)

    # Impute missing values using median grouped by gender
    for col in zero_cols:
        data[col] = data.groupby('gender')[col].transform(
            lambda x: x.fillna(x.median())
        )

    ##### Feature Engineering #####

    # Calculate BMI
    data['BMI'] = data['weight_kg'] / ((data['height_cm']/100) ** 2)

    # Calculate strength-to-weight ratio
    data['strength_to_weight'] = data['gripForce'] / data['weight_kg']

    # Calculate flexibility-to-height ratio
    data['flexibility_ratio'] = data['sit and bend forward_cm'] / data['height_cm']

    # Create age groups (decade-wise)
    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]
    )

    # 3. Convert categorical variables
    # Convert gender to numeric
    data['gender_numeric'] = data['gender'].map({'M': 1, 'F': 0})

    # 4. Prepare X and y
    # Define target variable
    y = data['class']

    # Convert target to numeric (A=0, B=1, C=2, D=3)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Define feature set (drop the original categorical columns and target)
    X = data.drop(['class', 'gender', 'age_group'], axis=1)

    # 5. Split the data into train, validation, and test sets
    # First split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size),
        random_state=random_state, stratify=y_trainval
    )

    # 6. Normalize/standardize features
    # Define the columns to standardize
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    # Fit the scaler on the training data only
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Standardize numeric columns using the training data statistics
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # Use the same scaler (fit on training data) for validation and test sets
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'label_encoder': label_encoder,
        'class_mapping': class_mapping,
        'scaler': scaler,
        'feature_names': list(X_train.columns)
    }


def load_and_preprocess_data(file_path, **kwargs):
    df = pd.read_csv(file_path)
    return preprocess_body_performance_data(df, **kwargs)


if __name__ == "__main__":
    # Example usage
    import os

    # Check if the file exists in the current directory
    if os.path.exists("bodyPerformance.csv"):
        preprocessed_data = load_and_preprocess_data("bodyPerformance.csv")

        print(f"Preprocessed data shapes:")
        print(f"X_train: {preprocessed_data['X_train'].shape}")
        print(f"X_val: {preprocessed_data['X_val'].shape}")
        print(f"X_test: {preprocessed_data['X_test'].shape}")

        print(f"\nClass mapping: {preprocessed_data['class_mapping']}")

        print("\nFeatures after preprocessing:")
        for i, feature in enumerate(preprocessed_data['feature_names']):
            print(f"{i+1}. {feature}")
    else:
        print("Dataset file not found. Please place the bodyPerformance.csv file in the current directory.")
