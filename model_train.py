# ================================
# 1. IMPORTS
# ================================
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ================================
# 2. LOAD DATA
# ================================
with open('power_spectrum_10000.pkl', 'rb') as f:
    data = pickle.load(f)

X = data["power_spectrum"]
y = data["label"]


# ================================
# 3. TRAIN-TEST SPLIT (FIXED)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # IMPORTANT
    random_state=42
)


# ================================
# 4. MODEL PIPELINE (BEST PRACTICE)
# ================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# ================================
# 5. TRAIN MODEL
# ================================
pipeline.fit(X_train, y_train)


# ================================
# 6. EVALUATION (CORRECT WAY)
# ================================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Test Accuracy:", accuracy)
print("Test AUC:", auc)


# ================================
# 7. CROSS-VALIDATION (VERY IMPORTANT)
# ================================
cv_scores = cross_val_score(pipeline, X, y, cv=5)

print("Cross-validation Accuracy:", cv_scores.mean())


# ================================
# 8. SAVE MODEL (FOR WEBSITE)
# ================================
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved as model.pkl")
