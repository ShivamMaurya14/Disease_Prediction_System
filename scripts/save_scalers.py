
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle

print("--- Generating and Saving Scalers ---\n")

def save_diabetes_scaler():
    print("1. Processing Diabetes Dataset...")
    try:
        df = pd.read_csv('datasets/diabetes.csv')
        
        # Reproduce Preprocessing Steps from Final_Diabetes_Prediction.ipynb
        
        # 1. Handle 0 values
        cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
        
        # 2. KNN Imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        df[cols_to_fix] = knn_imputer.fit_transform(df[cols_to_fix])
        
        # 3. Outlier Capping (Winsorization)
        def cap_outliers(df, col):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low_lim = Q1 - 1.5 * IQR
            up_lim = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < low_lim, low_lim, df[col])
            df[col] = np.where(df[col] > up_lim, up_lim, df[col])
            return df

        for col in ['Insulin', 'DiabetesPedigreeFunction']:
            df = cap_outliers(df, col)
            
        # 4. Feature Engineering: BMI Categories
        def categorize_bmi(bmi):
            if bmi < 18.5: return 0
            elif 18.5 <= bmi < 25: return 1
            elif 25 <= bmi < 30: return 2
            else: return 3

        df['BMI_Cat'] = df['BMI'].apply(categorize_bmi)
        
        X = df.drop('Outcome', axis=1)
        
        # Fit Scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save Scaler
        pickle.dump(scaler, open('diabetes_scaler.pkl', 'wb'))
        print("   ✅ Saved 'diabetes_scaler.pkl'")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def save_heart_scaler():
    print("\n2. Processing Heart Disease Dataset...")
    try:
        df = pd.read_csv('datasets/heart.csv')
        X = df.drop('target', axis=1)
        
        # Fit Scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save Scaler
        pickle.dump(scaler, open('heart_scaler.pkl', 'wb'))
        print("   ✅ Saved 'heart_scaler.pkl'")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    save_diabetes_scaler()
    save_heart_scaler()
