import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from db import save_prediction

# ======================================================
# MODEL PATHS  (✅ FIXED)
# ======================================================
stage1_model_path = "models/cancer_stage1_model.keras"
stage2_model_path = "models/multi_cancer_stage2_model.keras"

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
IMG_SIZE = (224, 224)

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
# IMAGE PREPROCESSING
# ======================================================
def preprocess_image(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_stage1_model():
    return tf.keras.models.load_model(stage1_model_path)

@st.cache_resource
def load_stage2_model():
    return tf.keras.models.load_model(stage2_model_path)

stage1_model = load_stage1_model()
stage2_model = load_stage2_model()

# ======================================================
# STAGE-1 INFERENCE
# ======================================================
def cancer_presence_screening(img):
    x = preprocess_image(img)
    preds = stage1_model.predict(x, verbose=0)[0]

    if preds.shape[0] == 1:
        conf = float(preds[0])
        return conf >= 0.5, conf
    else:
        idx = np.argmax(preds)
        return idx == 1, float(preds[idx])

# ======================================================
# STAGE-2 INFERENCE
# ======================================================
def cancer_type_classification(img):
    x = preprocess_image(img)
    probs = stage2_model.predict(x, verbose=0)[0]
    idx = np.argmax(probs)
    return STAGE2_CLASSES[idx], float(probs[idx]), probs

# ======================================================
# MAIN LAYOUT
# ======================================================
main_col, right_col = st.columns([3, 0.5])

with main_col:
    st.markdown("## IntelliScan: AI-Assisted Cancer Screening System")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Medical Image", ["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

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
                save_prediction(uploaded_file.name, False, conf1, None, None)

        if st.session_state.get("cancer_detected", False):
            st.markdown("### Step 2: Cancer Type Screening")

            if st.button("Run Type Screening"):
                cancer_type, conf2, probs = cancer_type_classification(image)

                st.session_state["stage2_result"] = {
                    "type": cancer_type,
                    "confidence": conf2,
                    "probs": probs
                }

                st.success(f"🧬 **Detected Cancer Type:** {cancer_type}")
                save_prediction(
                    uploaded_file.name,
                    True,
                    st.session_state["presence_conf"],
                    cancer_type,
                    conf2
                )

# ======================================================
# AI DIAGNOSIS SUMMARY
# ======================================================
if "stage2_result" in st.session_state:
    st.markdown("---")
    st.markdown("## 🧠 AI Diagnosis Summary")

    r = st.session_state["stage2_result"]
    probs = np.array(r["probs"]).flatten()   # ✅ FIX

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

    df_probs = pd.DataFrame({
        "Cancer Type": STAGE2_CLASSES,
        "Probability": probs
    }).set_index("Cancer Type")

    st.bar_chart(df_probs, use_container_width=True)
    st.warning("⚠️ AI-assisted screening only. Not a clinical diagnosis.")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("AI-assisted screening only. Not a substitute for medical diagnosis.")
st.markdown(
    "<p style='text-align:center;font-size:11px;color:#9e9e9e;'>"
    "Academic Medical AI Project | CDAC"
    "</p>",
    unsafe_allow_html=True
)