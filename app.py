
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set Page Config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 1.2rem;
    }
    .main-header {
        font-size: 3.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #306998;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem !important;
        padding: 0.8rem !important;
    }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-size: 1.2rem !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_diabetes_model():
    model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('models/diabetes_scaler.pkl', 'rb'))
    return model, scaler

@st.cache_resource
def load_heart_model():
    model = pickle.load(open('models/heart_model.pkl', 'rb'))
    scaler = pickle.load(open('models/heart_scaler.pkl', 'rb'))
    return model, scaler

@st.cache_resource
def load_xray_model():
    # Suppress TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    return load_model('models/chest_xray_model.h5')

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.title("Disease Prediction")
# --- Session State Logic ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

def set_page(page_name):
    st.session_state['page'] = page_name

# --- Sidebar Navigation (Hidden for Custom Nav) ---
# We are using custom buttons on Home, but let's keep a sidebar for direct access if needed
with st.sidebar:
    selected = st.selectbox("Navigate", ["Home", "Diabetes", "Heart Disease", "Chest X-Ray", "Brain Tumor Detection"])
    if selected != st.session_state['page']:
        st.session_state['page'] = selected
        st.rerun()

# --- Navbar / Header ---
if st.session_state['page'] != 'Home':
    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("🏠 Home"):
            set_page('Home')
            st.rerun()
    with col2:
        st.write("") # Spacer

# --- Home Page ---
if st.session_state['page'] == 'Home':
    st.markdown("<div class='main-header'>Disease Prediction System</div>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;'>
        <h3 style='color: #31333F;'>AI-Powered Early Diagnosis & Health Monitoring</h3>
        <p style='color: #555;'>Leveraging state-of-the-art Machine Learning models to provide accurate, real-time health risk assessments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Available Modules")
    
    # Cards Layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FF4B4B; margin-bottom: 10px;'>
            <h3 style='color: #FF4B4B;'>🩸 Diabetes</h3>
            <p>Predict diabetes risk based on diagnostic measures like Glucose, BMI, and Insulin levels.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Diagnose Diabetes"):
            set_page('Diabetes')
            st.rerun()
        
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #4B8BBE; margin-bottom: 10px;'>
            <h3 style='color: #4B8BBE;'>❤️ Heart Disease</h3>
            <p>Assess heart disease probability using cardiac metrics like Chest Pain, Cholesterol, and ECG.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Diagnose Heart Disease"):
            set_page('Heart Disease')
            st.rerun()

    with col3:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FFD700; margin-bottom: 10px;'>
            <h3 style='color: #E6C200;'>🩻 Chest X-Ray</h3>
            <p>Detect Pneumonia from chest X-Ray images using raw pixel analysis and Deep Learning.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Chest X-Ray"):
            set_page('Chest X-Ray')
            st.rerun()

    # New Row for Brain Tumor
    col1_2, col2_2, col3_2 = st.columns(3)
    with col1_2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #8E44AD; margin-bottom: 10px;'>
            <h3 style='color: #8E44AD;'>🧠 Brain Tumor</h3>
            <p>Detect Brain Tumors from MRI scans using Transfer Learning (Xception).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze MRI"):
            set_page('Brain Tumor Detection')
            st.rerun()
        
    st.markdown("---")

# --- Diabetes Page ---
if st.session_state['page'] == "Diabetes":
    st.markdown("<div class='main-header'>Diabetes Prediction</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        
    with col2:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        
    if st.button("Predict Diabetes Risk"):
        try:
            model, scaler = load_diabetes_model()
            
            # Feature Engineering: BMI Categories
            def categorize_bmi_val(bmi_val):
                if bmi_val < 18.5: return 0
                elif 18.5 <= bmi_val < 25: return 1
                elif 25 <= bmi_val < 30: return 2
                else: return 3
                
            bmi_cat = categorize_bmi_val(bmi)
            
            # Create input array
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, bmi_cat]])
            
            # Scale
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction[0] == 1:
                st.markdown(f"<div class='result-box danger'>High Risk of Diabetes Detected (Probability: {probability:.2f})</div>", unsafe_allow_html=True)
                st.warning("Please consult a healthcare professional.")
            else:
                st.markdown(f"<div class='result-box safe'>Low Risk of Diabetes (Probability: {probability:.2f})</div>", unsafe_allow_html=True)
                st.balloons()
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- Heart Disease Page ---
if st.session_state['page'] == "Heart Disease":
    st.markdown("<div class='main-header'>Heart Disease Prediction</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_h = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        if sex == "Male": sex_val = 1
        else: sex_val = 0
            
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        # Encoding check: dataset usually 0,1,2,3
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_val = cp_map[cp]
        
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        fbs_val = 1 if fbs == "True" else 0
        
    with col2:
        restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
        restecg_val = restecg_map[restecg]
        
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
        
        oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=0.0)
        slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2} # verification might be needed on exact mapping
        slope_val = slope_map[slope]
        
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3} # Common mapping, check dataset if different
        thal_val = thal_map[thal]
        
    if st.button("Predict Heart Disease Risk"):
        try:
            model, scaler = load_heart_model()
            
            input_data = np.array([[age_h, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            
            # The scaler expects 13 features
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            if prediction[0] == 1:
                st.markdown(f"<div class='result-box danger'>High Risk of Heart Disease Detected (Probability: {probability:.2f})</div>", unsafe_allow_html=True)
                st.warning("Please consult a cardiologist.")
            else:
                st.markdown(f"<div class='result-box safe'>Low Risk of Heart Disease (Probability: {probability:.2f})</div>", unsafe_allow_html=True)
                st.balloons()
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- Chest X-Ray Page ---
if st.session_state['page'] == "Chest X-Ray":
    st.markdown("<div class='main-header'>Chest X-Ray Pneumonia Detection</div>", unsafe_allow_html=True)
    
    st.write("Upload a chest X-Ray image to detect if it shows signs of Pneumonia.")
    
    uploaded_file = st.file_uploader("Choose an X-Ray Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption='Uploaded X-Ray', use_container_width=True)
            
            if st.button("Analyze X-Ray"):
                model = load_xray_model()
                
                # Preprocess Image
                img = Image.open(uploaded_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                # Predict
                prediction = model.predict(img_array)
                probability = prediction[0][0]
                
                # Threshold usually 0.5
                if probability > 0.5:
                    st.markdown(f"<div class='result-box danger'>PNEUMONIA Detected (Confidence: {probability:.2%})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box safe'>NORMAL (Confidence: {1-probability:.2%})</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")

# --- Brain Tumor Detection Page ---
if st.session_state['page'] == "Brain Tumor Detection":
    st.markdown("<div class='main-header'>Brain Tumor Detection</div>", unsafe_allow_html=True)
    st.write("Upload an MRI scan to detect brain tumors.")

    # Check if model exists
    model_path = 'models/brain_tumor_model.h5'
    if not os.path.exists(model_path):
        st.error("⚠️ Model not found!")
        st.warning("Please run the training script to generate the model: `python scripts/train_brain_tumor.py`")
    else:
        uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
            
            if st.button("Analyze MRI"):
                try:
                    # Load model
                    # Suppress TF logs
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                    model = load_model(model_path)
                    
                    # Preprocess
                    img = image.resize((299, 299))
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0 # Rescale as per training
                    
                    # Predict
                    prediction = model.predict(img_array)
                    class_indices = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
                    class_names = list(class_indices.keys())
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"**Prediction:** {predicted_class.capitalize()}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                    
                    # Interpretation
                    if predicted_class == 'notumor':
                        st.balloons()
                        st.write("🎉 No tumor detected. The MRI looks healthy.")
                    else:
                        st.warning(f"⚠️ Potential {predicted_class} tumor detected. Please consult a specialist.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

# --- Footer ---
st.markdown("---")

