import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from db import save_prediction


stage1_model_path = "models/swin_cancer_stage1.pth"
stage2_model_path = "models/swin_cancer_stage2.pth"

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="IntelliScan: AI-Assisted Cancer Screening System",
    page_icon="🧬",
    layout="wide"
)

# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

STAGE1_CLASSES = ["Normal", "Cancer"]
STAGE2_CLASSES = ["Breast Cancer", "Kidney Cancer", "Lung Cancer", "Oral Cancer"]

# ======================================================
# LEFT SIDEBAR – Workflow & Guidelines
# ======================================================
with st.sidebar:
    st.markdown("## 🧭 Application Workflow")
    st.markdown(
        """
        1️⃣ Upload a medical image  
        2️⃣ Run cancer presence screening  
        3️⃣ Run cancer type screening  
        4️⃣ Review AI diagnosis summary  
        """
    )

    st.markdown("---")

    st.markdown("## 🏥 Medical Guidelines")
    st.markdown(
        """
        • AI output is **decision support only**  
        • False positives are possible  
        • Clinical validation is mandatory  
        • Academic & research use only
        """
    )

    st.markdown("---")

    st.markdown("## 🖼 Image Details")
    if "uploaded_image_meta" in st.session_state:
        meta = st.session_state["uploaded_image_meta"]
        st.markdown(
            f"""
            **Filename:** {meta['name']}  
            **Resolution:** {meta['size'][0]} × {meta['size'][1]}  
            **Mode:** {meta['mode']}
            """
        )
    else:
        st.caption("Upload an image to view details.")

# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(img):
    return transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_stage1_model():
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(stage1_model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

@st.cache_resource
def load_stage2_model():
    model = models.swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, 4)
    model.load_state_dict(torch.load(stage2_model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

stage1_model = load_stage1_model()
stage2_model = load_stage2_model()

# ======================================================
# INFERENCE
# ======================================================
@torch.no_grad()
def cancer_presence_screening(img):
    probs = torch.softmax(stage1_model(preprocess_image(img)), dim=1)[0]
    idx = torch.argmax(probs).item()
    return idx == 1, float(probs[idx])

@torch.no_grad()
def cancer_type_classification(img):
    probs = torch.softmax(stage2_model(preprocess_image(img)), dim=1)[0]
    idx = torch.argmax(probs).item()
    return STAGE2_CLASSES[idx], float(probs[idx]), probs.cpu().numpy()

# ======================================================
# MAIN LAYOUT (CENTER + RIGHT PANEL)
# ======================================================
main_col, right_col = st.columns([3, 0.5])

with main_col:
    st.markdown("## IntelliScan: AI-Assisted Cancer Screening System")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Medical Image", ["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state["uploaded_image_meta"] = {
            "name": uploaded_file.name,
            "size": image.size,
            "mode": image.mode
        }

        st.image(image, width=200)
        st.markdown("---")

        if st.button("Run Screening"):
            cancer_present, conf1 = cancer_presence_screening(image)
            st.session_state["cancer_detected"] = cancer_present
            st.session_state["presence_conf"] = conf1

            if cancer_present:
                st.error("❌ Cancer Detected")
            else:
                st.success("✅ No Cancer Detected")

        if st.session_state.get("cancer_detected", False):
            st.markdown("### Step 2: Cancer Type Screening")

            if st.button("Run Type Screening"):
                cancer_type, conf2, probs = cancer_type_classification(image)

                st.session_state["stage2_result"] = {
                    "type": cancer_type,
                    "confidence": conf2,
                    "probs": probs
                }

                st.success(f"**Detected Cancer Type:** {cancer_type}")

                save_prediction(
                    uploaded_file.name,
                    True,
                    st.session_state["presence_conf"],
                    cancer_type,
                    conf2
                )

# ==================================================
# AI DIAGNOSIS SUMMARY (ONLY AFTER TYPE PREDICT)
# ==================================================
if "stage2_result" in st.session_state:

    st.markdown("---")
    st.markdown("## 🧠 AI Diagnosis Summary")

    r = st.session_state["stage2_result"]

    st.markdown(
        f"""
        **Cancer Presence:** ❌ Detected  
        **Cancer Type:** **{r['type']}**  
        **Risk Level:** <span style="color:#e53935;font-weight:600;">High</span>  
        **Confidence:** Strong (≈ {r['confidence']*100:.0f}%)
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Class Probabilities")

    import pandas as pd
    df_probs = pd.DataFrame({
        "Cancer Type": STAGE2_CLASSES,
        "Probability": r["probs"]
    }).set_index("Cancer Type")

    st.bar_chart(df_probs, use_container_width=True)

    st.warning("⚠️ AI-assisted screening only. Not a clinical diagnosis.")