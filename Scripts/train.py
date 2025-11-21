# ==========================
# 1) IMPORT LIBRARIES
# ==========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ==========================
# 2) LOAD DATA
# ==========================
data_path = "../Data/heart.csv"
df = pd.read_csv(data_path)

print("Data Loaded Successfully ✔️")
print(df.head())

# ==========================
# 3) CHECK MISSING VALUES
# ==========================
print("\nChecking for missing values...")
print(df.isnull().sum())

# If missing values exist → fill or drop
df.fillna(df.mean(), inplace=True)

# ==========================
# 4) SPLIT FEATURES & TARGET
# ==========================
X = df.drop("target", axis=1)   # Features
y = df["target"]                # Target

# ==========================
# 5) TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ==========================
# 6) FEATURE SCALING
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 7) TRAINING THE MODEL
# ==========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel Training Completed ✔️")

# ==========================
# 8) EVALUATION
# ==========================
y_pred = model.predict(X_test_scaled)

print("\n📌 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================
# 9) SAVE MODEL & SCALER
# ==========================
os.makedirs("../Models", exist_ok=True)
joblib.dump(model, "../Models/heart_model.pkl")
joblib.dump(scaler, "../Models/scaler.pkl")

print("\nModel and scaler saved successfully to /models/ folder ✔️")
