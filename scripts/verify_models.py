import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Construct absolute path to models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def verify_diabetes_model():
    print("Verifying Diabetes Model...")
    try:
        model_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'diabetes_scaler.pkl')
        
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        # Dummy input: 8 features + 1 engineered feature = 9 features
        # pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, bmi_cat
        input_data = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31, 2]]) 
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        print(f"Diabetes Model Verified! Prediction: {prediction}")
    except Exception as e:
        print(f"Error verifying Diabetes Model: {e}")

def verify_heart_model():
    print("\nVerifying Heart Disease Model...")
    try:
        model_path = os.path.join(MODELS_DIR, 'heart_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'heart_scaler.pkl')
        
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        # Dummy input: 13 features
        # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        input_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        print(f"Heart Disease Model Verified! Prediction: {prediction}")
    except Exception as e:
        print(f"Error verifying Heart Disease Model: {e}")

def verify_xray_model():
    print("\nVerifying Chest X-Ray Model...")
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model_path = os.path.join(MODELS_DIR, 'chest_xray_model.h5')
        
        model = load_model(model_path)
        
        # Dummy image input: (1, 224, 224, 3)
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(input_data)
        print(f"Chest X-Ray Model Verified! Prediction shape: {prediction.shape}, Value: {prediction[0][0]}")
    except Exception as e:
        print(f"Error verifying Chest X-Ray Model: {e}")

def verify_brain_tumor_model():
    print("\nVerifying Brain Tumor Model...")
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        model_path = os.path.join(MODELS_DIR, 'brain_tumor_model.h5')
        
        if not os.path.exists(model_path):
            print(f"Brain Tumor Model not found at {model_path}. Please run scripts/train_brain_tumor.py")
            return

        model = load_model(model_path)
        
        # Dummy image input: (1, 299, 299, 3)
        input_data = np.random.rand(1, 299, 299, 3).astype(np.float32)
        prediction = model.predict(input_data)
        print(f"Brain Tumor Model Verified! Prediction shape: {prediction.shape}, Max Prob: {np.max(prediction):.4f}")
    except Exception as e:
        print(f"Error verifying Brain Tumor Model: {e}")

if __name__ == "__main__":
    verify_diabetes_model()
    verify_heart_model()
    verify_xray_model()
    verify_brain_tumor_model()
