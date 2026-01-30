# 🧬 IntelliScan: AI-Assisted Cancer Screening System

IntelliScan is a **two-stage deep learning–based medical image screening system** designed for **academic and research purposes**.  
The system performs **cancer presence detection** followed by **cancer type classification**, mimicking a real-world clinical screening workflow.

---

## 🚀 Project Overview

Traditional single-step cancer classification models often suffer from unnecessary complexity and false positives.  
IntelliScan addresses this by adopting a **two-stage pipeline**:

1. **Stage-1:** Detects whether cancer is present (screening)
2. **Stage-2:** Identifies the specific cancer type only if cancer is detected

This modular approach improves **interpretability**, **efficiency**, and **deployment flexibility**.

---

## 🧠 Methodology (Two-Stage Pipeline)

### 🔹 Stage 1 – Cancer Presence Screening
- Binary classification: `Normal` vs `Cancer`
- Acts as a **screening gate**
- Prevents unnecessary multi-class prediction

### 🔹 Stage 2 – Cancer Type Classification
- Activated only when cancer is detected
- Multi-class classification of cancer type
- Provides class probabilities and confidence score

---

## 📊 Dataset Summary

### Stage-1 Dataset (Binary Classification)
**Objective:** Detect cancer presence

**Classes:**
- Normal (Benign)
- Cancer (Malignant)

**Task:** Binary image classification

---

### Stage-2 Dataset (Multi-Class Classification)
**Objective:** Identify cancer type

**Classes:**
1. Breast Cancer  
2. Kidney Cancer  
3. Lung Cancer  
4. Oral Cancer  

**Task:** Multi-class image classification

---

### Dataset Structure

dataset/
├── stage1/
│   ├── normal/
│   └── cancer/
│
└── stage2/
    ├── breast/
    ├── kidney/
    ├── lung/
    └── oral/


---

## 🖼️ Image Preprocessing

- Resize images to **224 × 224**
- Convert to RGB
- Normalize pixel values (0–1)
- ImageNet-compatible normalization (for Swin models)

---

## 🧰 Tech Stack

### Machine Learning
- PyTorch (Swin Transformer)
- TensorFlow / Keras
- NumPy

### Frontend & Deployment
- Streamlit (Interactive UI)
- Python 3.10+

### Database
- MySQL (Prediction logging)

---

## 🖥️ Application Features

- Two-stage AI inference pipeline
- Interactive Streamlit interface
- Confidence-based predictions
- Class probability visualization (bar chart)
- Prediction logging to database
- Separate Torch & Keras inference apps

---

## 📁 Project Structure

multiclass-cancer-classification/
├── models/
│   ├── swin_cancer_stage1.pth
│   ├── swin_cancer_stage2.pth
│   ├── cancer_stage1_model.keras
│   └── multi_cancer_stage2_model.keras
│
├── torch.py          # PyTorch inference app
├── keras_app.py      # Keras inference app
├── db.py             # Database utilities
├── requirements.txt
└── README.md


---

## ▶️ How to Run

### 🔹 PyTorch Version

streamlit run torch.py

### 🔹 Keras Version

streamlit run keras_app.py
