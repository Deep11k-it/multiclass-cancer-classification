# 🧠 Multiclass Cancer Classification System

## 📌 Project Summary
This project implements a hierarchical deep learning–based cancer classification system using medical images. The system first predicts the main cancer type and then performs type-specific sub-classification to determine clinical severity such as benign, malignant, normal, or tumor.

Convolutional Neural Networks (CNNs) are used for image classification, while scalable data pipelines and PySpark-based analysis support efficient handling of large datasets. The trained models are used to perform predictions on unseen images, enabling structured and interpretable cancer diagnosis.

---

## 📊 Dataset Summary
The dataset used in this project consists of medical images categorized into multiple cancer types along with their respective sub-classes. It is organized in a structured directory format with separate training and testing folders.

### 🧬 Cancer Types Included
- **Breast Cancer**
- **Kidney Cancer**
- **Lung Cancer**
- **Oral Cancer**

### 🏷️ Sub-Class Information
Each cancer type contains type-specific sub-classes representing clinical conditions such as:
- Benign
- Malignant
- Normal
- Tumor
- Cancerous

> Sub-class definitions vary across cancer types; therefore, a hierarchical classification approach is adopted to avoid label ambiguity.

### 📦 Dataset Details
- **Data Type:** Medical Images (JPG/PNG)
- **Approximate Size:** ~5 GB
- **Data Split:** Train / Test

### ⚙️ Processing & Storage Notes
- Images are loaded in batches using TensorFlow data pipelines
- PySpark is used for dataset analysis and class distribution checks
- The dataset is excluded from this repository using `.gitignore` due to size constraints
