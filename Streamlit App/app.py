import streamlit as st
from model_helper import predict

st.title("Car Damage Detection")

uploaded_file = st.file_uploader("Upload the File or Photo",type=["jpg", "jpeg","png","bmp","tiff", "tif","webp"])

if uploaded_file:
    image_pth="temp_file.jpg"
    with open(image_pth,"wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption="Uploaded File", width="stretch")
        predicted=predict(image_pth)
        st.info(f"Predicted information :{predicted}")
