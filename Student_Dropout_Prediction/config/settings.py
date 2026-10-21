import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, ASSETS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "train.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

APP_TITLE = "Student Dropout Prediction System"
APP_ICON = "🎓"

NUMERICAL_FEATURES = [
    "Age_at_Enrollment",
    "CGPA_1st_Sem",
    "CGPA_2nd_Sem",
    "Attendance_Rate",
    "Active_Backlogs",
    "Family_Income_LPA"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Scholarship_Holder",
    "Tuition_Fees_Up_To_Date",
    "Hostel_Resident",
    "Counseling_Attended"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Dropout"