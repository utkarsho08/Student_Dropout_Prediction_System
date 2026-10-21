import joblib
import pandas as pd
from config.settings import (
    BEST_MODEL_PATH, LR_MODEL_PATH, RF_MODEL_PATH, XGB_MODEL_PATH, ALL_FEATURES
)
from utils.preprocessor import preprocess_data
from utils.logger import get_logger

logger = get_logger("Predictor")

MODEL_MAP = {
    "Best Performing Model": BEST_MODEL_PATH,
    "Random Forest": RF_MODEL_PATH,
    "XGBoost": XGB_MODEL_PATH,
    "Logistic Regression": LR_MODEL_PATH
}

def predict_dropout(input_dict: dict, model_choice: str = "Best Performing Model"):
    model_path = MODEL_MAP.get(model_choice, BEST_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError("Models not found. Please run training/train_models.py first.")
        
    model = joblib.load(model_path)
    df_input = pd.DataFrame([input_dict])[ALL_FEATURES]
    df_processed = preprocess_data(df_input, is_training=False)
    
    prediction = int(model.predict(df_processed)[0])
    probability = float(model.predict_proba(df_processed)[0][1])
    
    logger.info(f"Model: {model_choice} | Prediction: {prediction} | Risk Prob: {probability:.3f}")
    return prediction, probability