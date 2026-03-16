import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

def main(input_dir, output_dir, max_depth, n_estimators):
    train_path = os.path.join(input_dir, "train.csv")
    test_path = os.path.join(input_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']
    
    model = RandomForestClassifier(
        max_depth=max_depth, 
        n_estimators=n_estimators, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model trained. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    
    metrics = {"accuracy": float(acc), "f1": float(f1)}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing prepared data")
    parser.add_argument("output_dir", type=str, help="Directory to save artifacts")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the tree")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.max_depth, args.n_estimators)