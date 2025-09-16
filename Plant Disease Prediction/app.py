# app.py
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# ============================
# Paths
# ============================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "final_model.h5"
LABELS_PATH = BASE_DIR / "class_labels.npy"

# ============================
# Verify files exist
# ============================
if not MODEL_PATH.is_file():
    st.error(f"❌ Could not find model file at: {MODEL_PATH}")
    st.stop()
if not LABELS_PATH.is_file():
    st.error(f"❌ Could not find labels file at: {LABELS_PATH}")
    st.stop()

# ============================
# Load model + labels
# ============================
try:
    model = load_model(str(MODEL_PATH).replace("\\", "/"))
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

try:
    class_labels = list(np.load(str(LABELS_PATH), allow_pickle=True))
except Exception as e:
    st.error(f"❌ Failed to load labels: {e}")
    st.stop()

# ============================
# Disease remedies
# ============================
disease_info = {
    "Apple Scab": "🍎 Remove infected leaves, use fungicide sprays, keep orchard floor clean.",
    "Potato Early Blight": "🥔 Crop rotation, remove infected plants, apply fungicides.",
    "Tomato Leaf Spot": "🍅 Remove affected leaves, maintain proper spacing, apply copper-based fungicide.",
    "Healthy": "✅ Plant is healthy! Maintain water, sunlight, and nutrients."
}

# ============================
# Session state
# ============================
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "preds" not in st.session_state:
    st.session_state.preds = None  # Store prediction probabilities

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Plant Disease Predictor 🌱", layout="wide")
st.markdown("<h1 style='text-align:center; color:green'>🌱 Plant Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload a leaf image to predict disease and get remedies!</p>", unsafe_allow_html=True)

# File uploader
if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

# Display uploaded image and prediction
if st.session_state.uploaded_file:
    try:
        img = Image.open(st.session_state.uploaded_file).convert("RGB")
        width, height = img.size

        if width < 50 or height < 50:
            st.error("❌ Image too small or invalid. Please upload a proper leaf image.")
        else:
            st.image(img, caption="Uploaded Leaf", use_column_width=True)

            # Predict button
            if st.button("Predict Disease 🌿") and not st.session_state.prediction_done:
                with st.spinner("Predicting..."):
                    img_resized = img.resize((224, 224))
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    st.session_state.preds = model.predict(x)  # Save probabilities
                    class_idx = np.argmax(st.session_state.preds)
                    st.session_state.prediction_result = class_labels[class_idx]
                    st.session_state.confidence = np.max(st.session_state.preds)
                    st.session_state.prediction_done = True

            # Display prediction results only if prediction done
            if st.session_state.prediction_done:
                st.markdown(f"""
                <div style='border:2px solid #4CAF50; padding:15px; border-radius:10px; background-color:#f0fff0'>
                <h3 style='color:#4CAF50;'>✅ Prediction: {st.session_state.prediction_result}</h3>
                <p><strong>Confidence:</strong> {st.session_state.confidence*100:.2f}%</p>
                <p><strong>Remedy / Info:</strong> {disease_info.get(st.session_state.prediction_result, "No info available.")}</p>
                </div>
                """ , unsafe_allow_html=True)

                # Probability chart
                fig, ax = plt.subplots(figsize=(4,3))
                ax.barh(class_labels, st.session_state.preds[0], color="green")
                ax.set_xlabel("Probability")
                ax.set_title("Prediction Probabilities")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Could not process image. Upload a valid leaf. Error: {e}")

# Clear / reset button
if st.session_state.uploaded_file:
    if st.button("🗑️ Clear Image / New Prediction"):
        # Clear everything safely
        st.session_state.uploaded_file = None
        st.session_state.prediction_done = False
        st.session_state.prediction_result = None
        st.session_state.confidence = None
        st.session_state.preds = None
