#!/usr/bin/env bash
set -e

echo "===================================================="
echo " Starting Student College Dropout Prediction System "
echo "===================================================="

if [ ! -d "venv" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[2/3] Activating environment and verifying packages..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/3] Launching application on http://localhost:8501 ..."
python3 launch.py