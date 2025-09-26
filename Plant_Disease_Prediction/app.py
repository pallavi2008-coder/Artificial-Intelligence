import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import webbrowser

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_plant_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Class labels (order must match training!)
# -------------------------------
class_labels = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy",
}

# -------------------------------
# Expanded solutions with sources
# -------------------------------

from solutions import solutions

# -------------------------------
# Leaf detection function
# -------------------------------
def is_leaf(img_path, threshold=0.02):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size
    return green_ratio > threshold

# -------------------------------
# Prediction function
# -------------------------------
def predict_disease(img_path):
    if not is_leaf(img_path):
        return "❌ Error: Uploaded image does not appear to be a leaf.", None

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_name = class_labels[class_idx]
    return class_name, solutions.get(class_name)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")

st.title("🌱 Smart Plant Disease Prediction System")
st.write("Upload a leaf image to check if it's **healthy** or **diseased**, and get remedies with trusted resources.")

uploaded_file = st.file_uploader("📤 Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("temp.jpg", caption="🌿 Uploaded Leaf", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analyzing the leaf..."):
            prediction, solution = predict_disease("temp.jpg")

        if prediction.startswith("❌"):
            st.error(prediction)
        else:
            st.success(f"**Prediction:** {prediction}")

            if solution:
                st.subheader("🩺 Diagnosis & Remedies")
                st.write(f"**Summary:** {solution['summary']}")
                st.info(f"💡 Remedy: {solution['remedy']}")
                st.markdown(f"🔗 [Learn more here]({solution['source']})")
            else:
                st.warning("⚠️ No detailed remedy available for this disease yet.")


