import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ======================================================
# CONFIG (FREE TIER SAFE)
# ======================================================
DEVICE = torch.device("cpu")
IMG_SIZE = 224
MODEL_PATH = "models/swin_cancer_stage1.pth"

CLASSES = ["Normal", "Cancer"]

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="IntelliScan – Cancer Screening",
    page_icon="🧬",
    layout="centered"
)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## 🧭 Workflow")
    st.markdown(
        """
        1️⃣ Upload medical image  
        2️⃣ AI screens for cancer presence  
        3️⃣ Review confidence score  
        """
    )

    st.markdown("---")

    st.markdown("## ⚠️ Disclaimer")
    st.markdown(
        """
        • AI-assisted screening only  
        • Not a medical diagnosis  
        • Academic & research use  
        """
    )

# ======================================================
# IMAGE TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(img):
    return transform(img.convert("RGB")).unsqueeze(0)

# ======================================================
# LOAD MODEL (ONLY ONCE)
# ======================================================
@st.cache_resource
def load_model():
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ======================================================
# INFERENCE
# ======================================================
@torch.no_grad()
def predict(img):
    x = preprocess(img)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return CLASSES[idx], float(probs[idx])

# ======================================================
# MAIN UI
# ======================================================
st.markdown("## 🧬 IntelliScan – AI Cancer Screening")
st.markdown("Single-stage Torch deployment (Free Tier optimised)")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=220)

    if st.button("Run Screening"):
        label, confidence = predict(image)

        st.markdown("---")

        if label == "Cancer":
            st.error("❌ **Cancer Detected**")
        else:
            st.success("✅ **No Cancer Detected**")

        st.markdown(
            f"""
            **Prediction:** `{label}`  
            **Confidence:** `{confidence * 100:.2f}%`
            """
        )

        st.warning(
            "⚠️ This is an AI-assisted screening result. "
            "Clinical validation is required."
        )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Academic AI Project | IntelliScan")