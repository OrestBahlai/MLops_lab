import os
import json
import pandas as pd

def test_data_schema_basic():
    data_path = os.getenv("DATA_PATH", "data/prepared/train.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"
    
    df = pd.read_csv(data_path)
    
    assert "Churn" in df.columns, "Missing target column 'Churn'"
    
    assert df["Churn"].notna().all(), "Target column contains NaNs"
    
    assert df.shape[0] >= 50, "Too few rows for training"

def test_artifacts_exist():
    assert os.path.exists("models/model.pkl"), "model.pkl not found"
    assert os.path.exists("models/metrics.json"), "metrics.json not found"
    assert os.path.exists("models/confusion_matrix.png"), "confusion_matrix.png not found"

def test_quality_gate_f1():
    threshold = float(os.getenv("F1_THRESHOLD", "0.55")) 
    
    with open("models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    f1 = float(metrics.get("f1", 0))
    
    assert f1 >= threshold, f"Quality Gate failed: F1 score {f1:.4f} is lower than threshold {threshold:.2f}"