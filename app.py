import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
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

    st.title("Digit Recognition Sketchboard")
    st.markdown("Choose to draw a digit or upload an image, and click 'Predict' to see the model's prediction.")

    # Sidebar for instructions and settings
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
        1. Choose to either draw a digit or upload an image.
        2. Draw a digit (0-9) on the canvas or upload an image.
        3. Click the 'Predict' button to see the model's prediction.
        4. The predicted digit and the confidence score will be displayed.
    """)
    st.sidebar.title("Settings")
    stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 10)
    st.sidebar.markdown("Adjust the stroke width using the slider above.")

    # Option to choose between drawing and uploading
    option = st.sidebar.selectbox("Choose input method", ["Draw", "Upload"])

    if option == "Draw":
        # Create a canvas for drawing
        canvas = st_canvas(
            fill_color="white",
            stroke_width=stroke_width,
            stroke_color="black",
            background_color="white",
            height=canvas_height,
            width=canvas_width,
            key="canvas",
        )

        if st.button("Predict"):
            if canvas.image_data is not None:
                drawn_image = canvas.image_data
                preprocessed_image = preprocess_image(drawn_image)
                prediction, confidence = predict_digit(drawn_image)

                # Display the preprocessed image
                st.subheader("Preprocessed Image")
                st.image(preprocessed_image.reshape(28, 28), width=100, caption="Model Input Image", clamp=True)

                # Display the prediction and confidence
                st.subheader("Prediction")
                st.write(f"Predicted Digit: **{prediction}**")
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
            else:
                st.error("Please draw a digit on the canvas.")

    elif option == "Upload":
        st.markdown("""
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #d3d3d3;
            text-align: center;
            margin-bottom: 30px;
        }
        .file-upload {
            font-size: 16px;
            font-weight: bold;
            color: #d3d3d3;
            text-align: center;
            margin-top: 30px;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #d3d3d3;
            text-align: center;
            margin-top: 30px;
        }
        .download-button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #d3d3d3;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .download-button:hover {
            background-color: #555555;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.markdown('<div class="title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
        st.markdown("Upload an image of a handwritten digit (JPEG or PNG format) and let the model predict the digit.")

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
