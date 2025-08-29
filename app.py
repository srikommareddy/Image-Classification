#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ------------------------------
# Load the trained model
# ------------------------------
@st.cache_resource  # cache so it doesn’t reload every time
def load_cifar10_model():
    model = load_model("cifar10_cnn.h5")  # adjust path if needed
    return model

model = load_cifar10_model()

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🚀 CIFAR-10 Image Classification")
st.write("Upload an image and the trained CNN will predict its class.")

# ------------------------------
# Upload Image
# ------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ------------------------------
    # Preprocess Image
    # ------------------------------
    img_resized = image.resize((32, 32))  # CIFAR10 size
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # ------------------------------
    # Prediction
    # ------------------------------
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.markdown(f"### 🔮 Prediction: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {confidence:.2f}")

