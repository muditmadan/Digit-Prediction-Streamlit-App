import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import base64

# Load the model
model = tf.keras.models.load_model('final.h5')

# Define the canvas size and stroke width
canvas_width = 300
canvas_height = 300
stroke_width = 10

# Define a function to preprocess the image for the model
def preprocess_image(img):
    img = cv2.resize(255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255
    return img

# Define a function to get a prediction from the model
def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

# Create the Streamlit app
def main():
    st.set_page_config(page_title="Digit Recognition Sketchboard", page_icon="✏️", layout="centered", initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #444444;
            text-align: center;
            margin-bottom: 30px;
        }
        .instructions, .settings, .upload {
            font-size: 16px;
            font-weight: bold;
            color: #444444;
            text-align: left;
            margin-top: 30px;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #444444;
            text-align: center;
            margin-top: 30px;
        }
        .download-button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #0073e6;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            text-align: center;
            margin-top: 20px;
        }
        .download-button:hover {
            background-color: #005bb5;
        }
        .sidebar .element-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .canvas-container {
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Digit Recognition Sketchboard</div>', unsafe_allow_html=True)
    st.markdown("Choose to draw a digit or upload an image, and click 'Predict' to see the model's prediction.")

    # Sidebar for instructions and settings
    st.sidebar.markdown('<div class="instructions">Instructions</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        1. Choose to either draw a digit or upload an image.
        2. Draw a digit (0-9) on the canvas or upload an image.
        3. Click the 'Predict' button to see the model's prediction.
        4. The predicted digit and the confidence score will be displayed.
    """)
    st.sidebar.markdown('<div class="settings">Settings</div>', unsafe_allow_html=True)
    stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 10)
    st.sidebar.markdown("Adjust the stroke width using the slider above.")

    # Option to choose between drawing and uploading
    option = st.sidebar.selectbox("Choose input method", ["Draw", "Upload"])

    if option == "Draw":
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="black",
            background_color="white",
            height=canvas_height,
            width=canvas_width,
            key="canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Add "Predict Now" button
        if st.button('Predict Now'):
            if canvas_result.image_data is not None:
                input_numpy_array = np.array(canvas_result.image_data)
                input_image = cv2.cvtColor(input_numpy_array.astype('uint8'), cv2.COLOR_RGBA2BGR)
                digit, confidence = predict_digit(input_image)
                st.markdown(f'<div class="result">Predicted Digit: <span style="font-size: 48px;">{digit}</span> with confidence {confidence:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result">Please draw a digit on the canvas.</div>', unsafe_allow_html=True)

    elif option == "Upload":
        st.markdown('<div class="upload">Upload an image of a handwritten digit (JPEG or PNG format) and let the model predict the digit.</div>', unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            # Open the image
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image
            image_array = np.array(image)
            image_array = image_array / 255.0
            image_array = image_array.reshape((1, 28, 28, 1))

            # Predict the digit
            prediction = model.predict(image_array)
            predicted_label = np.argmax(prediction)

            # Display the predicted digit
            st.markdown(f'<div class="result">Predicted Digit: <span style="font-size: 48px;">{predicted_label}</span></div>', unsafe_allow_html=True)

            # Create a downloadable text file with the predicted digit
            text_file = f"Predicted Digit: {predicted_label}"
            b64 = base64.b64encode(text_file.encode()).decode()
            href = f'<a class="download-button" href="data:file/txt;base64,{b64}" download="predicted_digit.txt">Download Result</a>'
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.markdown('<div class="file-upload">Please upload an image file.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
