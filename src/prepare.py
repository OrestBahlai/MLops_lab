import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

def prepare_data(input_file, output_dir):

    df = pd.read_csv(input_file)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    df = df.drop('customerID', axis=1)
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Data prepared and saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare.py <input_file> <output_dir>")
        sys.exit(1)
        
    input_file_path = sys.argv[1]
    output_dir_path = sys.argv[2]
    
    prepare_data(input_file_path, output_dir_path)