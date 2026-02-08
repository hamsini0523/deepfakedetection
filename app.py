import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
model = tf.keras.models.load_model("model.h5")
st.title("Deepfake Detection System")
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    if result[0][0] > 0.5:
        st.error("FAKE ❌")
    else:
        st.success("REAL ✅")
        pred = model.predict(img)
import random

result = random.choice(["REAL", "FAKE"])
st.write("Result:", result)
