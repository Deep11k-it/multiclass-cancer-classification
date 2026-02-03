# IntelliScan: AI-Assisted Cancer Screening System

IntelliScan is a **two-stage deep learning–based medical image screening system** designed for **academic and research purposes**.  
The system performs **cancer presence detection** followed by **cancer type classification**, mimicking a real-world clinical screening workflow.

---
**Presentation Link:**  
https://www.canva.com/design/DAG_7ZOU4tM/0WaPoyyRuGOmtpWVfkwPCQ/view?utm_content=DAG_7ZOU4tM&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h1f382a4ec7#18

---

## Project Overview

Traditional single-step cancer classification models often suffer from unnecessary complexity and false positives.  
IntelliScan addresses this by adopting a **two-stage pipeline**:

1. **Stage-1:** Detects whether cancer is present (screening)
2. **Stage-2:** Identifies the specific cancer type only if cancer is detected

This modular approach improves **interpretability**, **efficiency**, and **deployment flexibility**.

---

##  Methodology (Two-Stage Pipeline)

### 🔹 Stage 1 – Cancer Presence Screening
- Binary classification: `Normal` vs `Cancer`
- Acts as a **screening gate**
- Prevents unnecessary multi-class prediction

### 🔹 Stage 2 – Cancer Type Classification
- Activated only when cancer is detected
- Multi-class classification of cancer type
- Provides class probabilities and confidence score

---
### Trained Model Link:

**Stage 1:**
https://drive.google.com/file/d/1dJftQV07qX3qlxfY-EpjOAYJsIsXJ6P7/view?usp=drive_link

**Stage 2:**
https://drive.google.com/file/d/1tKP2LyxF18k1GRzPm08pEGfmIoNh2r_g/view?usp=drive_link

---

# Dataset Summary

In this project, we used publicly available **medical image datasets from Kaggle** to build a **two-stage cancer prediction system**.  
The datasets are used only for **academic and learning purposes**.

The idea behind using two datasets is to first **detect whether cancer is present** and then **identify the cancer type only if cancer is detected**.

---

## Datasets Used

### 1. Cancer Prediction – Stage 1  
**Kaggle Link:**  
https://www.kaggle.com/datasets/yuvrajkari7/cancer-prediction-stage1

**Purpose:**  
This dataset is used for **Stage 1 screening**, where the model predicts whether an image indicates **cancer or not**.

**Classes:**
- Normal (Benign)
- Cancer (Malignant)

This stage acts as a **filter**, so that only cancer-positive images are passed to the next stage.

---

### 2. Multi-Cancer Prediction – Stage 2  
**Kaggle Link:**  
https://www.kaggle.com/datasets/yuvrajkari7/multi-cancer-prediction-stage-2

**Purpose:**  
This dataset is used for **Stage 2 classification**, where the model predicts the **type of cancer**.

**Classes:**
- Breast Cancer  
- Kidney Cancer  
- Lung Cancer  
- Oral Cancer  

This stage is executed **only when Stage 1 predicts cancer**.

---

## Dataset Structure
```
dataset/
├── stage1/
│   ├── train/
│   │   ├── normal/
│   │   └── cancer/
│   └── test/
│       ├── normal/
│       └── cancer/
│
├── stage2/
│   ├── train/
│   │   ├── breast/
│   │   ├── kidney/
│   │   ├── lung/
│   │   └── oral/
│   └── test/
│       ├── breast/
│       ├── kidney/
│       ├── lung/
│       └── oral/

```
Each folder contains medical images belonging to that class.

---

## Image Details

- Image format: RGB images  
- Image size: Varies across dataset  
- Images are resized to a fixed size during preprocessing  
- Datasets may contain class imbalance, which is common in medical data

---

## Preprocessing Performed

Before training the models, the following steps were applied:

- Image resizing (e.g., 224×224)
- Pixel normalization
- Removal of corrupted images
- Data augmentation:
  - Rotation
  - Horizontal flip
  - Zoom

Only **safe augmentations** were used to avoid altering important medical patterns.

---

## Evaluation Considerations

Since this is a **medical screening problem**:
- **Recall** is treated as an important metric
- Reducing false negatives is prioritized
- Accuracy alone is not considered sufficient

---

## Disclaimer

This dataset and the trained models are intended **only for academic and educational purposes**.  
They are **not suitable for real medical diagnosis or clinical use**.

---

## Summary

- Two Kaggle medical image datasets were used
- Stage 1 performs cancer screening
- Stage 2 performs cancer type classification
- The dataset design supports a realistic medical workflow

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

## Application Features

- Two-stage AI inference pipeline
- Interactive Streamlit interface
- Confidence-based predictions
- Class probability visualization (bar chart)
- Prediction logging to database
- Separate Torch & Keras inference apps

---

## 📁 Project Structure

```
multiclass-cancer-classification/
├── __pycache__/                 # Python cache files
├── mccp_venv/                   # Virtual environment (local)
│
├── models/                      # Trained models
│   ├── cancer_stage1_model.keras
│   ├── multi_cancer_stage2_model.keras
│   ├── swin_cancer_stage1.pth
│   └── swin_cancer_stage2.pth
│
├── scripts/                     # Model training notebooks
│   ├── stage1.ipynb             # Stage 1 training (cancer screening)
│   └── stage2.ipynb             # Stage 2 training (cancer type classification)
│
├── .env                         # Environment variables (ignored in git)
├── .gitignore                   # Git ignore rules
│
├── cancer_predictions.xlsx      # Exported prediction results
├── db.py                        # Database connection and logging logic
│
├── EC2_deployment_steps.md      # AWS EC2 deployment instructions
├── EC2_stage1_torch.py          # Stage 1 inference (PyTorch)
├── EC2_torch_app.py             # Streamlit app (PyTorch)
│
├── export_predictions.py        # Export predictions from DB to file
├── intelliscan.pem              # EC2 key file (should NOT be committed)
│
├── keras_app.py                 # Streamlit app (TensorFlow / Keras)
├── torch_app.py                 # Streamlit app (PyTorch – local)
│
├── test.py                      # Testing / experimentation script
├── Requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```


---

## ▶️ How to Run

### 🔹 PyTorch Version

streamlit run torch.py

### 🔹 Keras Version

streamlit run keras_app.py



