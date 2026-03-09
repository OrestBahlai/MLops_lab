import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
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
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Experiment")
    
    with mlflow.start_run():
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        
        model = RandomForestClassifier(
            max_depth=max_depth, 
            n_estimators=n_estimators, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            "acc_train": accuracy_score(y_train, y_pred_train),
            "f1_train": f1_score(y_train, y_pred_train),
            "acc_test": accuracy_score(y_test, y_pred_test),
            "f1_test": f1_score(y_test, y_pred_test)
        }
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.pkl")
        joblib.dump(model, model_path)
        
        print(f"Model trained and saved to {model_path} | Test Acc: {metrics['acc_test']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str, help="Directory containing prepared data")
    parser.add_argument("output_dir", type=str, help="Directory to save the trained model")

    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the tree")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.max_depth, args.n_estimators)