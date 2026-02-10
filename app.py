
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import gdown

# Global Fix: Monkey-patch Keras layers to handle the 'list' error bug (Keras 3)
import tensorflow as tf
try:
    import keras
    classes_to_patch = [tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling2D]
    if hasattr(keras, 'layers'):
        if hasattr(keras.layers, 'Flatten'): classes_to_patch.append(keras.layers.Flatten)
        if hasattr(keras.layers, 'GlobalAveragePooling2D'): classes_to_patch.append(keras.layers.GlobalAveragePooling2D)
    
    def apply_patch(cls):
        original_call = cls.call
        def patched_call(self, inputs, *args, **kwargs):
            if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
                inputs = inputs[0]
            return original_call(self, inputs, *args, **kwargs)
        cls.call = patched_call

    for cls in set(classes_to_patch):
        apply_patch(cls)
except Exception as patch_err:
    pass

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
    diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
    diabetes_scaler = pickle.load(open('models/diabetes_scaler.pkl', 'rb'))
    return diabetes_model, diabetes_scaler

@st.cache_resource
def load_heart_model():
    heart_model = pickle.load(open('models/heart_model.pkl', 'rb'))
    heart_scaler = pickle.load(open('models/heart_scaler.pkl', 'rb'))
    return heart_model, heart_scaler

@st.cache_resource
def load_xray_model():
    """Load X-Ray model with error reporting"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    errors = []
    xray_path = 'models/xrays_pneumonia.keras'
    
    try:
        if os.path.exists(xray_path):
            # Use global patch for Flatten
            xray_model = tf.keras.models.load_model(xray_path, compile=False)
            return xray_model, None
        else:
            errors.append(f"Model file not found at {xray_path}")
    except Exception as e:
        errors.append(f"Load error: {str(e)}")
        try:
            # Try with safe_mode fallback
            xray_model = tf.keras.models.load_model(xray_path, compile=False, safe_mode=False)
            return xray_model, None
        except Exception as e2:
            errors.append(f"SafeMode fallback error: {str(e2)}")
            
    return None, errors

@st.cache_resource
def download_model_from_drive(file_id, output_path):
    """Download model from Google Drive using gdown"""
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        return True, None
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_brain_tumor_model():
    """Load brain tumor model with support for .keras, SavedModel, and H5 formats. 
    Downloads from Drive if not found locally."""
    # Note: Flatten patch is already applied globally at the top of the file
    errors = []
    
    file_id = "12oBWm5zYq7az62TPq7w68iFz5IOTygrG"
    keras_path = 'models/brain_tumor_model.keras'
    
    # Check if any local version exists
    savedmodel_path = 'models/brain_tumor_model'
    h5_path = 'models/brain_tumor_model.h5'
    
    if not (os.path.exists(keras_path) or os.path.exists(savedmodel_path) or os.path.exists(h5_path)):
        with st.status("📥 Downloading Brain Tumor Model from Google Drive..."):
            success, err = download_model_from_drive(file_id, keras_path)
            if not success:
                errors.append(f"Download error: {err}")
    
    try:
        if os.path.exists(keras_path):
            try:
                # Primary attempt: Standard load
                tumor_model = tf.keras.models.load_model(keras_path, compile=False)
                return tumor_model, None
            except Exception as e_inner:
                # Secondary attempt: Loading with safe_mode=False (for TF 2.16+)
                tumor_model = tf.keras.models.load_model(keras_path, compile=False, safe_mode=False)
                return tumor_model, None
    except Exception as e:
        errors.append(f".keras: {str(e)}")
    
    try:
        if os.path.exists(savedmodel_path):
            tumor_model = tf.keras.models.load_model(savedmodel_path, compile=False)
            return tumor_model, None
    except Exception as e:
        errors.append(f"SavedModel: {str(e)}")
    
    try:
        if os.path.exists(h5_path):
            tumor_model = tf.keras.models.load_model(h5_path, compile=False)
            return tumor_model, None
    except Exception as e:
        errors.append(f".h5 (standard): {str(e)}")
        try:
            tumor_model = tf.keras.models.load_model(h5_path, compile=False, safe_mode=False)
            return tumor_model, None
        except Exception as e2:
            errors.append(f".h5 (safe_mode=False): {str(e2)}")
            try:
                # Rebuild matching Xception architecture
                from tensorflow.keras.applications import Xception
                from tensorflow.keras.layers import Dense, Dropout, Flatten
                from tensorflow.keras.models import Sequential
                
                base_model = Xception(include_top=False, weights=None, input_shape=(299, 299, 3), pooling='max')
                
                tumor_model = Sequential([
                    base_model,
                    Flatten(), # Use standard Flatten (it's patched)
                    Dropout(rate=0.3),
                    Dense(128, activation='relu'),
                    Dropout(rate=0.25),
                    Dense(4, activation='softmax')
                ])
                
                # Try to load weights
                # Weights might be in H5 or .keras (as weights)
                if os.path.exists(keras_path):
                    tumor_model.load_weights(keras_path)
                else:
                    tumor_model.load_weights(h5_path)
                return tumor_model, None
            except Exception as e3:
                errors.append(f"Weights-only: {str(e3)}")
                
    return None, errors

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
    page_options = ["Home", "Diabetes", "Heart Disease", "Chest X-Ray", "Brain Tumor Detection"]
    current_index = page_options.index(st.session_state['page']) if st.session_state['page'] in page_options else 0
    selected = st.selectbox("Navigate", page_options, index=current_index)
    if selected != st.session_state['page']:
        st.session_state['page'] = selected
        # Clear results on page change
        for key in ['res_diabetes', 'res_heart', 'res_xray', 'res_mri']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Navbar / Header ---
if st.session_state['page'] != 'Home':
    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("🏠 Home", key="btn_home"):
            st.session_state['page'] = 'Home'
            # Clear results
            for key in ['res_diabetes', 'res_heart', 'res_xray', 'res_mri']:
                if key in st.session_state:
                    del st.session_state[key]
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
    
    st.markdown("### Choose the Disease You Want to Check")
    
    # Cards Layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FF4B4B; margin-bottom: 10px;'>
            <h4 style='color: #FF4B4B;'>🩸 Diabetes</h4>
            <p style='font-size: 0.9rem;'>Predict risk using Glucose, BMI, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check Diabetes", key="btn_diabetes", width="stretch"):
            st.session_state['page'] = 'Diabetes'
            st.rerun()
        
    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #4B8BBE; margin-bottom: 10px;'>
            <h4 style='color: #4B8BBE;'>❤️ Heart</h4>
            <p style='font-size: 0.9rem;'>Assess risk using cardiac metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check Heart", key="btn_heart", width="stretch"):
            st.session_state['page'] = 'Heart Disease'
            st.rerun()

    with col3:
        st.markdown("""
        <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #FFD700; margin-bottom: 10px;'>
            <h4 style='color: #E6C200;'>🩻 X-Ray</h4>
            <p style='font-size: 0.9rem;'>Detect Pneumonia from X-Rays.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check X-Ray", key="btn_xray", width="stretch"):
            st.session_state['page'] = 'Chest X-Ray'
            st.rerun()

    with col4:
        st.markdown("""
        <div style='background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #8E44AD; margin-bottom: 10px;'>
            <h4 style='color: #8E44AD;'>🧠 Tumor</h4>
            <p style='font-size: 0.9rem;'>Detect Tumors from MRI scans.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check MRI", key="btn_mri", width="stretch"):
            st.session_state['page'] = 'Brain Tumor Detection'
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
        
    if st.button("Predict Diabetes Risk", key="btn_predict_diabetes"):
        try:
            diabetes_model, diabetes_scaler = load_diabetes_model()
            
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
            input_scaled = diabetes_scaler.transform(input_data)
            
            # Predict
            prediction = diabetes_model.predict(input_scaled)
            probability = diabetes_model.predict_proba(input_scaled)[0][1]
            
            st.session_state['res_diabetes'] = {
                'pred': prediction[0],
                'prob': probability
            }
            st.success("Analysis Complete!")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
    if 'res_diabetes' in st.session_state:
        res = st.session_state['res_diabetes']
        if res['pred'] == 1:
            st.markdown(f"<div class='result-box danger'>High Risk of Diabetes Detected (Probability: {res['prob']:.2f})</div>", unsafe_allow_html=True)
            st.warning("Please consult a healthcare professional.")
        else:
            st.markdown(f"<div class='result-box safe'>Low Risk of Diabetes (Probability: {res['prob']:.2f})</div>", unsafe_allow_html=True)
            st.balloons()

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
        
    if st.button("Predict Heart Disease Risk", key="btn_predict_heart"):
        try:
            heart_model, heart_scaler = load_heart_model()
            
            input_data = np.array([[age_h, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            
            # The scaler expects 13 features
            input_scaled = heart_scaler.transform(input_data)
            
            prediction = heart_model.predict(input_scaled)
            probability = heart_model.predict_proba(input_scaled)[0][1]
            
            st.session_state['res_heart'] = {
                'pred': prediction[0],
                'prob': probability
            }
            st.success("Analysis Complete!")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    if 'res_heart' in st.session_state:
        res = st.session_state['res_heart']
        if res['pred'] == 1:
            st.markdown(f"<div class='result-box danger'>High Risk of Heart Disease Detected (Probability: {res['prob']:.2f})</div>", unsafe_allow_html=True)
            st.warning("Please consult a cardiologist.")
        else:
            st.markdown(f"<div class='result-box safe'>Low Risk of Heart Disease (Probability: {res['prob']:.2f})</div>", unsafe_allow_html=True)
            st.balloons()

# --- Chest X-Ray Page ---
if st.session_state['page'] == "Chest X-Ray":
    st.markdown("<div class='main-header'>Chest X-Ray Pneumonia Detection</div>", unsafe_allow_html=True)
    
    st.write("Upload a chest X-Ray image to detect if it shows signs of Pneumonia.")
    
    uploaded_file = st.file_uploader("Choose an X-Ray Image", type=["jpg", "png", "jpeg"])
    
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded X-Ray', width="stretch")
        
        if st.button("Analyze X-Ray", key="btn_analyze_xray", width="stretch"):
            try:
                with st.spinner('Analyzing X-Ray...'):
                    # Clear previous result if any
                    if 'res_xray' in st.session_state:
                        del st.session_state['res_xray']
                        
                    xray_model, xray_errors = load_xray_model()
                    
                    if xray_model is None:
                        st.error("⚠️ **X-Ray Model Loading Error**")
                        if xray_errors:
                            with st.expander("🔍 Technical Details (Loading failed)"):
                                for err in xray_errors:
                                    st.code(err)
                        st.stop() # Stop the analysis here
                    
                    # Preprocess Image
                    img = Image.open(uploaded_file)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    img = img.resize((224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0
                    
                    # Predict
                    prediction = xray_model.predict(img_array)
                    
                    # Handle different output shapes (binary vs categorical)
                    if len(prediction[0]) > 1:
                        # Assume categorical (e.g. [normal_prob, pneumonia_prob])
                        probability = prediction[0][1] 
                    else:
                        # Assume binary sigmoid output
                        probability = prediction[0][0]
                    
                    st.session_state['res_xray'] = float(probability)
                    st.success("Analysis Complete!")
                    st.rerun()

            except Exception as e:
                import traceback
                st.error(f"Error during analysis: {e}")
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

    if 'res_xray' in st.session_state:
        prob = st.session_state['res_xray']
        if prob > 0.5:
            st.markdown(f"<div class='result-box danger'>PNEUMONIA Detected (Confidence: {prob:.2%})</div>", unsafe_allow_html=True)
            st.warning("⚠️ The scan shows signs of pneumonia. Please consult a doctor.")
        else:
            st.markdown(f"<div class='result-box safe'>NORMAL (Confidence: {1-prob:.2%})</div>", unsafe_allow_html=True)
            st.success("✅ The scan appears normal.")

# --- Brain Tumor Detection Page ---
if st.session_state['page'] == "Brain Tumor Detection":
    st.markdown("<div class='main-header'>Brain Tumor Detection</div>", unsafe_allow_html=True)
    st.write("Upload an MRI scan to detect brain tumors.")

    # The model will be automatically downloaded from Google Drive if missing locally 
    # during the analysis phase (using load_brain_tumor_model)
    
    uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img, caption="Uploaded MRI", width="stretch")
        
        if st.button("Analyze MRI", key="btn_analyze_mri", width="stretch"):
            with st.spinner('Analyzing MRI scan...'):
                # Load model using cached function
                tumor_model, loading_errors = load_brain_tumor_model()
                
                if tumor_model is None:
                    st.error("⚠️ **Model Loading Error**")
                    st.warning("""
                    The brain tumor model could not be loaded even from the new `.keras` file.
                    
                    **To fix this:**
                    1. Ensure the file `models/brain_tumor_model.keras` is not corrupted.
                    2. If you just retrained the model, make sure you saved it using `model.save('models/brain_tumor_model.keras')`.
                    """)
                    if loading_errors:
                        with st.expander("🔍 What does this mean? (Technical Details)"):
                            st.write("The app tried multiple ways to load your model, but they all failed with these messages:")
                            for err in loading_errors:
                                st.code(err)
                else:
                    try:
                        # Preprocess
                        img = uploaded_img.resize((299, 299))
                        img_array = np.array(img)
                        
                        # Handle grayscale images
                        if len(img_array.shape) == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]
                        
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array / 255.0 # Rescale as per training
                        
                        # Predict
                        prediction = tumor_model.predict(img_array)
                        class_indices = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
                        class_names = list(class_indices.keys())
                        predicted_class = class_names[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100
                        
                        st.session_state['res_mri'] = {
                            'class': predicted_class,
                            'conf': confidence
                        }
                        st.success("Analysis Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        import traceback
                        st.error(f"Error during analysis: {e}")
                        with st.expander("Show detailed error"):
                            st.code(traceback.format_exc())

    if 'res_mri' in st.session_state:
        res = st.session_state['res_mri']
        st.success(f"**Prediction:** {res['class'].capitalize()}")
        st.info(f"**Confidence:** {res['conf']:.2f}%")
        
        # Interpretation
        if res['class'] == 'notumor':
            st.balloons()
            st.write("🎉 No tumor detected. The MRI looks healthy.")
        else:
            st.warning(f"⚠️ Potential {res['class']} tumor detected. Please consult a specialist.")

# --- Footer ---
st.markdown("---")
