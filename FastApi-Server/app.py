import streamlit as st
import requests
from PIL import Image
import io

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"   # change when deploying backend

st.set_page_config(
    page_title="Image Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Image Prediction System")
st.write("Upload an image to get prediction from the AI model.")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        with st.spinner("Sending image to model..."):

            try:

                # Convert image to bytes
                img_bytes = uploaded_file.getvalue()

                files = {
                    "file": (
                        uploaded_file.name,
                        img_bytes,
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:

                    result = response.json()

                    prediction = result.get("Prediction", "No prediction")

                    st.success("Prediction Complete ✅")
                    st.subheader(f"Prediction: {prediction}")

                else:
                    st.error("Server Error")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI server")

            except Exception as e:
                st.error(f"Error: {str(e)}")