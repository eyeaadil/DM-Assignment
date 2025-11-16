import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Task 1: Use Real-World Dataset ---
print("--- Loading Titanic Dataset ---")
df = sns.load_dataset('titanic')

# Drop 'deck' due to too many missing values and 'who', 'adult_male', 'alive' as they are redundant
df = df.drop(['deck', 'who', 'adult_male', 'alive', 'class', 'embark_town'], axis=1)

print(df.head())
print("\n")

# --- Data Preprocessing ---
print("--- Preprocessing Data ---")

# Define features (X) and target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Identify numerical and categorical features
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked']

# Create preprocessing pipelines for both data types
# This is a robust way to handle preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fill missing age/fare with median
    ('scaler', StandardScaler()) # Scale data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing embarked
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Convert categories to numbers
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into training and testing sets BEFORE fitting
# This is crucial for a valid evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
print("\n")


# --- Task 2: Apply Random Forest & XGBoost ---

# --- Model 1: Random Forest ---
print("--- Training Random Forest (Baseline) ---")
# Create a full pipeline including preprocessing and the model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Train the model
rf_pipeline.fit(X_train, y_train)

# Evaluate the baseline model
y_pred_rf = rf_pipeline.predict(X_test)
print("Random Forest - Baseline Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\n")

# --- Model 2: XGBoost ---
print("--- Training XGBoost (Baseline) ---")
# Note: You might need to run: pip install xgboost
# Create the XGBoost pipeline
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
# Train the model
xgb_pipeline.fit(X_train, y_train)

# Evaluate the baseline model
y_pred_xgb = xgb_pipeline.predict(X_test)
print("XGBoost - Baseline Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
# 
print(confusion_matrix(y_test, y_pred_xgb))
print("\n")


# --- Task 3: Tune and Evaluate Model ---
print("--- Tuning Random Forest Model ---")
# We will tune the Random Forest model (rf_pipeline)

# Define the parameters to search
# We test different numbers of trees and tree depths
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create the GridSearch object
# cv=5 means 5-fold cross-validation
# n_jobs=-1 means use all available CPU cores
grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

# Run the search
print("Running GridSearchCV... (This may take a minute)")
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_

print("\n--- Tuned Model Evaluation ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")

# Evaluate the FINAL tuned model on the TEST set
y_pred_tuned = best_rf_model.predict(X_test)
print("\nFinal Tuned Random Forest - Test Set Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(classification_report(y_test, y_pred_tuned))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\n")


# --- Task 4: Optional Deployment (Save Model) ---
model_filename = 'best_titanic_model.joblib'
print(f"--- Saving Best Model to {model_filename} ---")
# We save the 'best_rf_model' which is the entire tuned pipeline
joblib.dump(best_rf_model, model_filename)

print("Script complete. All tasks implemented.")