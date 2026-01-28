import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
from db import save_prediction   # ✅ DB import (ONLY ONCE)

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="IntelliScan: AI-Assisted Cancer Screening",
    page_icon="🧬",
    layout="centered"
)

# ======================================================
# Configuration
# ======================================================
IMG_SIZE = (224, 224)

CANCER_TYPES = [
    "Breast Cancer",
    "Kidney Cancer",
    "Lung Cancer",
    "Oral Cancer"
]

# ======================================================
# Load Cancer Type Model (Optional)
# ======================================================
@st.cache_resource
def load_type_model():
    model_path = os.path.join("models", "cancer_type_model.keras")

    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None   # 👈 model not available yet

type_model = load_type_model()

# ======================================================
# Image Preprocessing
# ======================================================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================================================
# Dummy Cancer Presence Screening (Stage 1)
# ======================================================
def cancer_presence_screening(img):
    time.sleep(1)
    return False, 0.91   # dummy output

# ======================================================
# Sidebar (LEFT PANEL)
# ======================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analytics Dashboard")

TABLEAU_DASHBOARD_URL = "https://public.tableau.com/app/profile/ashutosh.choudhary6027/viz/Montana_Brew_Coffee_Analysis/Dashboard1"

st.sidebar.markdown(
    f"""
    <a href="{TABLEAU_DASHBOARD_URL}" target="_blank">
        <button style="
            width:100%;
            padding:10px;
            border-radius:6px;
            border:none;
            background-color:#1e88e5;
            color:white;
            font-size:14px;
            cursor:pointer;">
            📊 View Prediction Dashboard
        </button>
    </a>
    """,
   unsafe_allow_html=True
)
st.sidebar.title("🩺 Clinical Decision Support")

st.sidebar.markdown(
    """
    **AI-assisted screening tool**  
    For academic & research use only
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 🔄 Workflow
    1. Upload medical image  
    2. Cancer presence screening  
    3. Cancer type classification  
    4. Confidence-based output  
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### 🧠 Model Architecture
    - CNN-based Deep Learning  
    - Trained on multi-cancer datasets  
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ⚠ **Disclaimer**  
    This system does **not** replace  
    professional medical diagnosis.
    """
)

# ======================================================
# Header
# ======================================================
st.markdown(
    "<h2 style='text-align:center;'>IntelliScan: AI-Assisted Cancer Screening System</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#9e9e9e;'>"
    "AI-based clinical decision support for cancer screening and classification"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================================
# Upload Image
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png"],
    key="uploaded_image"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(image, width=110)
    with col2:
        st.caption("Image uploaded successfully.")

    st.markdown("---")

    # ==================================================
    # STEP 1: Cancer Presence Screening
    # ==================================================
    st.subheader("Step 1: Cancer Presence Screening")

    if st.button("Run Screening"):
        with st.spinner("Performing initial screening..."):
            cancer_present, cancer_conf = cancer_presence_screening(image)

        if cancer_present:
            st.markdown(
                """
                <span style="
                    display:inline-block;
                    padding:6px 14px;
                    border-radius:20px;
                    background-color:#fdecea;
                    color:#b71c1c;
                    font-weight:600;">
                    ❌ Cancer Detected
                </span>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Advisory: AI screening indicates possible cancer-related patterns. "
                "Further clinical evaluation by a medical professional is recommended."
            )

            st.markdown(
                f"""
                <p style="color:#cfcfcf; margin-top:8px;">
                <b>Prediction Summary</b><br>
                Cancer Detected: <b style="color:#e53935;">Yes</b><br>
                Confidence: {cancer_conf * 100:.2f}%<br>
                Next Step: Type Classification
                </p>
                """,
                unsafe_allow_html=True
            )

            st.session_state["cancer_detected"] = True
            st.session_state["presence_conf"] = cancer_conf

        else:
            st.markdown(
                """
                <span style="
                    display:inline-block;
                    padding:6px 14px;
                    border-radius:20px;
                    background-color:#edf7ed;
                    color:#1b5e20;
                    font-weight:600;">
                    ✅ No Cancer Detected
                </span>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Advisory: No significant cancer indicators detected. "
                "Routine monitoring is suggested."
            )

            # ✅ Save NO-cancer result
            save_prediction(
                image_name=uploaded_file.name,
                cancer_present=False,
                presence_conf=float(cancer_conf),
                cancer_type=None,
                type_conf=None
            )

            st.session_state["cancer_detected"] = False

    # ==================================================
    # STEP 2: Cancer Type Classification
    # ==================================================
    if st.session_state.get("cancer_detected", False):

        st.markdown("---")

        if st.button("🧬 Classify Cancer Type"):
            with st.spinner("Classifying cancer type..."):
                img = preprocess_image(image)
                preds = type_model.predict(img)

                idx = np.argmax(preds)
                cancer_type = CANCER_TYPES[idx]
                confidence = preds[0][idx] * 100

            col1, col2 = st.columns(2)
            col1.metric("Cancer Type", cancer_type)
            col2.metric("Confidence", f"{confidence:.2f}%")

            # ✅ Save cancer + type result
            save_prediction(
                image_name=uploaded_file.name,
                cancer_present=True,
                presence_conf=float(st.session_state["presence_conf"]),
                cancer_type=cancer_type,
                type_conf=float(confidence / 100)
            )

            st.session_state["final_done"] = True

    # ==================================================
    # FINAL BUTTON: RESET
    # ==================================================
    if st.session_state.get("final_done", False):

        st.markdown("---")

        if st.button("🔄 Predict Next"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption(
    "AI-assisted screening only. This system does not provide a medical diagnosis. "
    "All results must be reviewed by qualified healthcare professionals."
)

st.markdown(
    "<hr><p style='text-align:center;font-size:11px;color:#9e9e9e;'>"
    "Academic Medical AI Project | CDAC"
    "</p>",
    unsafe_allow_html=True
)