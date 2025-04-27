import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import numpy as np

base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

base.trainable = False

model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


st.title("Skin Disease Prediction")

uploaded_file = st.file_uploader("Upload an image of the skin", type=["jpg", "jpeg", "png"])

class_names = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma", 
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis", 
    "squamous cell carcinoma",
    "vascular lesion",
]

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    print(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(predicted_class)

    # Display result
    st.success(f"Predicted Skin Disease: {class_names[predicted_class]}")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # image preprocessing
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # printing prediction
    st.success(f"Predicted Skin Disease: {class_names[predicted_class]}")
