import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from config.settings import NUMERICAL_FEATURES, SCALER_PATH

BINARY_MAP = {
    "Gender": {"Male": 1, "Female": 0, 1: 1, 0: 0},
    "Scholarship_Holder": {"Yes": 1, "No": 0, 1: 1, 0: 0},
    "Tuition_Fees_Up_To_Date": {"Yes": 1, "No": 0, 1: 1, 0: 0},
    "Hostel_Resident": {"Yes": 1, "No": 0, 1: 1, 0: 0},
    "Counseling_Attended": {"Yes": 1, "No": 0, 1: 1, 0: 0}
}

def preprocess_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    data = df.copy()
    
    for col, mapping in BINARY_MAP.items():
        if col in data.columns:
            data[col] = data[col].map(mapping).fillna(0).astype(int)
            
    if is_training:
        scaler = StandardScaler()
        data[NUMERICAL_FEATURES] = scaler.fit_transform(data[NUMERICAL_FEATURES])
        joblib.dump(scaler, SCALER_PATH)
    else:
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            data[NUMERICAL_FEATURES] = scaler.transform(data[NUMERICAL_FEATURES])
            
    return data