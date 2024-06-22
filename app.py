import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained MNIST model
model = tf.keras.models.load_model('MNISTModel.keras')

# Function to preprocess image for model prediction
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 pixels
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Reshape image to fit model input (add batch dimension)
    img = np.reshape(resized, (1, 28, 28, 1)) / 255.0
    return img

# Streamlit app
def main():
    st.title('MNIST Digit Recognizer')

    # Instructions for the user
    st.write("Draw a digit on the canvas or upload an image to classify it.")

    # Create a canvas component for drawing
    canvas_result = st_canvas(
        fill_color="black",  # Background color of the canvas
        stroke_width=20,      # Width of the stroke for drawing
        stroke_color="white", # Color of the stroke
        background_color="black",  # Background color of the component
        height=150,           # Height of the canvas
        width=150,            # Width of the canvas
        drawing_mode="freedraw",  # Drawing mode which can be "freedraw" or "transform"
        key="canvas"
    )

    # Check if the canvas is not empty
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale
        canvas_img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        # Resize to 28x28 pixels
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        # Reshape image to fit model input (add batch dimension)
        img = np.reshape(resized, (1, 28, 28, 1)) / 255.0

        # Predict button
        if st.button("Predict"):
            # Predict
            prediction = model.predict(img)
            digit = np.argmax(prediction)
            st.write(f"Prediction: {digit}")

    # File uploader for image
    st.write("---")
    uploaded_file = st.file_uploader("Or upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = np.array(Image.open(uploaded_file))
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        img_processed = preprocess_image(img)
        prediction = model.predict(img_processed)
        digit = np.argmax(prediction)
        st.write(f"Prediction: {digit}")

if __name__ == '__main__':
    main()
