import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATA_PATH, BEST_MODEL_PATH, LR_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
    METRICS_PATH, ASSETS_DIR, ALL_FEATURES, TARGET_COLUMN
)
from utils.preprocessor import preprocess_data
from utils.logger import get_logger

logger = get_logger("ModelTrainer")

def generate_synthetic_data(n_samples: int = 1500) -> pd.DataFrame:
    np.random.seed(42)
    age = np.random.randint(17, 25, n_samples)
    cgpa_1 = np.round(np.random.uniform(4.0, 10.0, n_samples), 2)
    cgpa_2 = np.round(np.clip(cgpa_1 + np.random.normal(0, 0.6, n_samples), 3.5, 10.0), 2)
    attendance = np.round(np.random.uniform(40.0, 100.0, n_samples), 1)
    backlogs = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.5, 0.2, 0.12, 0.08, 0.06, 0.04])
    income = np.round(np.random.uniform(1.5, 18.0, n_samples), 2)
    
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    scholarship = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    tuition_up_to_date = np.random.choice(["Yes", "No"], size=n_samples, p=[0.78, 0.22])
    hostel = np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    counseling = np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])

    # Calculate underlying dropout logit score
    risk_score = (
        -0.45 * cgpa_2
        - 0.04 * attendance
        + 0.65 * backlogs
        + np.where(tuition_up_to_date == "No", 1.4, 0.0)
        + np.where(scholarship == "No", 0.35, -0.25)
        - 0.08 * income
        + 3.8
    )
    prob = 1 / (1 + np.exp(-risk_score))
    dropout = (np.random.uniform(0, 1, n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "Age_at_Enrollment": age,
        "CGPA_1st_Sem": cgpa_1,
        "CGPA_2nd_Sem": cgpa_2,
        "Attendance_Rate": attendance,
        "Active_Backlogs": backlogs,
        "Family_Income_LPA": income,
        "Gender": gender,
        "Scholarship_Holder": scholarship,
        "Tuition_Fees_Up_To_Date": tuition_up_to_date,
        "Hostel_Resident": hostel,
        "Counseling_Attended": counseling,
        TARGET_COLUMN: dropout
    })
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Generated synthetic dataset at {DATA_PATH}")
    return df

def train_and_evaluate():
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else generate_synthetic_data()
    
    processed_df = preprocess_data(df, is_training=True)
    X = processed_df[ALL_FEATURES]
    y = processed_df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), LR_MODEL_PATH),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42), RF_MODEL_PATH),
        "XGBoost": (XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=5, eval_metric="logloss", random_state=42), XGB_MODEL_PATH)
    }
    
    metrics_summary = {}
    best_f1 = -1.0
    best_model_obj = None
    best_model_name = ""
    
    plt.figure(figsize=(8, 6))
    
    for name, (model, save_path) in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, save_path)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_summary[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_obj = model
            best_model_name = name
            
    # Save ROC Curve
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Student Dropout Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "roc_curve.png")
    plt.close()
    
    # Save Best Model
    joblib.dump(best_model_obj, BEST_MODEL_PATH)
    metrics_summary["Best_Model"] = best_model_name
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # Save Confusion Matrix of Best Model
    y_best_pred = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Dropout"], yticklabels=["Retained", "Dropout"])
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion_matrix.png")
    plt.close()
    
    # Save Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(processed_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "heatmap.png")
    plt.close()

    # Save Feature Importance (using Random Forest)
    rf_model = models["Random Forest"][0]
    importances = pd.Series(rf_model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#3b82f6")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "feature_importance.png")
    plt.close()

    print(f"Training Complete! Best Model: {best_model_name} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    train_and_evaluate()