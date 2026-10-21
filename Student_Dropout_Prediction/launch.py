import subprocess
import sys
from config.settings import BEST_MODEL_PATH

def main():
    if not BEST_MODEL_PATH.exists():
        print("Models not found. Running initial model training...")
        subprocess.run([sys.executable, "training/train_models.py"], check=True)
        
    print("Launching Streamlit server on localhost...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()