import streamlit as st
import numpy as np
import tensorflow as tf
import gdown
from PIL import Image

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="Crop Classification System", layout="centered")

st.title("🌱 Crop Classification System")
st.write("Upload crop image and choose model")

# ===========================
# DOWNLOAD MODELS FROM DRIVE
# ===========================
def download_model(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# ResNet
download_model("1LIMor8vr613Udfkyhn8_6rYECugPvdg5", "resnet_model.h5")

# MobileNet
download_model("1YFtplC4nQsgWFCEOsQoRgyVKq5zr1Hbf", "mobilenet_model.h5")

# ===========================
# LOAD MODELS
# ===========================
resnet_model = tf.keras.models.load_model("resnet_model.h5", compile=False)
mobilenet_model = tf.keras.models.load_model("mobilenet_model.h5", compile=False)

# ===========================
# CLASS LABELS
# ===========================
class_names = [
    "Cherry",
    "Coffee-plant",
    "Cucumber",
    "Lemon",
    "Pearl_millet(bajra)",
    "banana",
    "cotton",
    "jowar",
    "maize",
    "rice",
    "soyabean",
    "wheat"
]

# ===========================
# MODEL SELECTION
# ===========================
model_choice = st.selectbox("Choose Model", ["ResNet50", "MobileNetV2"])

# ===========================
# IMAGE UPLOAD
# ===========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    image = image.resize((224, 224))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # ===========================
    # PREPROCESSING FIX
    # ===========================
    if model_choice == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img_array = preprocess_input(img_array)
        prediction = resnet_model.predict(img_array)
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_array = preprocess_input(img_array)
        prediction = mobilenet_model.predict(img_array)

    # ===========================
    # RESULT
    # ===========================
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction Result")
    st.success(f"🌾 Crop: {pred_class}")
    st.info(f"🎯 Confidence: {confidence:.2f}")