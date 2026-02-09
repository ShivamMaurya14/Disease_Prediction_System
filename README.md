# Disease Prediction System 🏥

> **Thinking Machine Hackathon Submission** - *Indian Institute of Information Technology, Pune*  
> *"Revolutionizing Early Diagnostics with Artificial Intelligence"*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## �️ Video Demonstration

<!-- Embed your video here. You can use a YouTube link or a direct file path if hosted in the repo -->
[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE)

> *Click the image above to watch the full walkthrough of the application.*

*(Alternatively, place your video file in the repository and link it here: `[Download Demo Video](./video/demo.mp4)`)*

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

## 💻 Installation & Setup

### Prerequisites
-   Python 3.8 or higher
-   Git installed

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/disease-prediction-system.git
    cd disease-prediction-system
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train/Generate Model Files** ⚠️ **IMPORTANT**
    
    > **The `models/` directory is not included in the repository** due to GitHub's 100MB file size limit. You must generate the model files locally by running the training notebooks:
    
    ```bash
    # For Brain Tumor Detection (creates brain_tumor_model.h5 - ~242MB)
    # Open and run: notebooks/Final_Brain_Tumor_Prediction.ipynb
    
    # For other models, run their respective training notebooks:
    # - notebooks/Diabetes-Prediction-14414d.ipynb
    # - notebooks/Heart-Disease-Prediction.ipynb (if exists)
    ```
    
    All model files will be automatically saved to the `models/` directory after training.

5.  **Run the Application**
    ```bash
    streamlit run app.py
    # OR
    bash run_app.sh
    ```

6.  **Access the App**
    Open your browser and navigate to `http://localhost:8501`.

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


**Team Name**  
*Thinking Machine Hackathon 2024*

-   **Shivam Maurya** - *AI & ROBOTICS ENGINEER*
-   **[Saurav Tiwari]** - *Frontend & UI/UX*
-   **[Rishabh Panday]** - *Data Scientist/ML Engineer*

## Latest Updates
-   **New Notebook**: Added [`Final_Brain_Tumor_Prediction.ipynb`](notebooks/Final_Brain_Tumor_Prediction.ipynb) for Brain Tumor model training.
-   **New Module**: Brain Tumor Detection integrated into the app.

---

Made with ❤️ and ☕ for healthcare innovation.
