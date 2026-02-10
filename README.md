# Disease Prediction System 🏥

> **Thinking Machine Hackathon Submission** - *Indian Institute of Information Technology, Pune*  
> *"Revolutionizing Early Diagnostics with Artificial Intelligence"*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📽️ Video Demonstration

Experience a full walkthrough of the **MedSynapse Disease Prediction System** in action:

<div align="center">
  <video src="https://github.com/ShivamMaurya14/Disease_Prediction_System/raw/main/assets/demo.mp4" width="100%" autoplay loop muted playsinline></video>
</div>

> *The video covers MRI analysis, X-Ray detection, and tabular data predictions.*

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Future Scope](#-future-scope)
- [Contributors](#-contributors)

---

## 🚀 Overview

The **Disease Prediction System** is an integrated healthcare platform designed to assist medical professionals and individuals in the early detection of critical diseases. Leveraging the power of **Machine Learning** and **Deep Learning**, our application provides instant, accurate risk assessments for Diabetes, Heart Disease, and Pneumonia.

## 🎯 Problem Statement

Early diagnosis is crucial for effective treatment and management of chronic diseases. However, access to rapid diagnostics can be limited by:
-   **Cost**: Expensive medical tests.
-   **Time**: Long waiting periods for results.
-   **Accessibility**: Lack of specialists in remote areas.

Our goal is to bridge this gap by providing a **low-cost, AI-driven initial screening tool**.

## 💡 Solution Architecture

Our solution combines three predictive models into a unified, user-friendly interface:
1.  **Tabular Data Analysis**: Using Random Forest and Logistic Regression for numerical health records (Diabetes & Heart).
2.  **Computer Vision**: Using Convolutional Neural Networks (CNNs) for medical imaging (Chest X-Rays).
3.  **Interactive UI**: A Streamlit-based web app for seamless user interaction.

---

## 🌟 Key Features

### 🩸 Diabetes Prediction
*Predicts the likelihood of diabetes based on diagnostic measures.*
-   **Model**: Random Forest Classifier
-   **Accuracy**: ~98% (on test set)
-   **Inputs**: Glucose, BMI, Insulin, Age, Blood Pressure, etc.

### ❤️ Heart Disease Prediction
*Assesses cardiovascular health risks.*
-   **Model**: Logistic Regression / Random Forest
-   **Accuracy**: ~85%
-   **Inputs**: Chest Pain Type, Cholesterol, Max Heart Rate, ECG, etc.

### 🩻 Pneumonia Detection (X-Ray)
*Analyzes chest X-Rays to detect signs of Pneumonia.*
-   **Model**: Deep Learning CNN (VGG16/ResNet based)
-   **Inputs**: Chest X-Ray Images (JPEG/PNG)
-   **Output**: Normal vs. Pneumonia Classification logic.

### 🧠 Brain Tumor Detection (MRI)
*Detects brain tumors from MRI scans and classifies them.*
-   **Model**: Xception (Transfer Learning)
-   **Inputs**: Brain MRI Images
-   **Output**: Glioma, Meningioma, No Tumor, Pituitary classification.

---

## 🛠️ Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=Streamlit&logoColor=white) |
| **ML Models** | ![Scikit-Learn](https://img.shields.io/badge/-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) *Transfer Learning (Xception)* |
| **Environment** | ![Anaconda](https://img.shields.io/badge/-Anaconda-44A833?logo=anaconda&logoColor=white) ![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white) |

---

## 🌐 Live Access

The **MedSynapse Disease Prediction System** is deployed and ready for immediate use. You can access the fully functional application without any local setup via the link below:

### [👉 Launch Live Application](https://dps-medisynapse.streamlit.app/#team-medsynapse)

> **Note**: The live version handles all model loading and processing in the cloud, provided by Streamlit Cloud.

---

## 💻 Local Developer Setup (Optional)

If you wish to run the system locally for development purposes:

1. **Clone & Install**:
   ```bash
   git clone https://github.com/ShivamMaurya14/Disease_Prediction_System.git
   cd Disease_Prediction_System
   pip install -r requirements.txt
   ```

2. **Run App**:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit Application
├── models/               # Trained ML/DL Models (.pkl, .h5)
├── notebooks/            # Jupyter Notebooks for training
├── scripts/              # Helper Scripts (verification, utilities)
├── assets/               # Static assets (images, demo video)
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation
```

---

## 🧪 Model Verification

To verify that all models are present and loading correctly without starting the UI, run our verification script:

```bash
python scripts/verify_models.py
```

---

## � Future Scope
-   [ ] **Mobile App**: Develop a Flutter/React Native version for on-the-go access.
-   [ ] **More Diseases**: Add modules for Skin Cancer (Dermatology), Kidney Disease, and Liver Disease.
-   [ ] **Doctor Connect**: Feature to book appointments with specialists if high risk is detected.
-   [ ] **Report Generation**: Download a PDF report of the analysis.

---

## 👥 Contributors

**Team - MEDSYNAPSE**  
*Thinking Machine Hackathon 2026*


-   **Shivam Maurya** - *AI & Robotics Engineer*
-   **[Saurav Tiwari]** - *Frontend & Streamlit Developer*
-   **[Rishabh Panday]** - *Data Scientist/ML Engineer*

---

Made with ❤️ and ☕ for healthcare innovation.
