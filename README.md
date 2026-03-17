# 🚗 Car Damage Detection using Deep Learning

An end-to-end **Deep Learning + Streamlit application** that detects and classifies vehicle damage from images.

This project leverages **transfer learning (ResNet50)** to identify the **location and severity of car damage**, making it useful for real-world applications like insurance automation and vehicle inspection.

---
[![Live App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen?logo=streamlit)](https://car-damage-detection-using-deeplearning.streamlit.app/)

---

## 🌐 Live Demo

Click below to try the app:

👉 https://car-damage-detection-using-deeplearning.streamlit.app/

---

## 📌 Demo

Upload an image of a car, and the model will predict:

- Front Normal  
- Front Crushed  
- Front Breakage  
- Rear Normal  
- Rear Crushed  
- Rear Breakage  

> ⚠️ Note: The model performs best on **third-quarter front or rear view images**.

---

## 🧠 Model Overview

- **Architecture:** ResNet50 (Transfer Learning)
- **Dataset Size:** ~1700 images
- **Number of Classes:** 6
- **Validation Accuracy:** ~80%

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy & Pandas
- Streamlit

---

## 🚀 Features

- Image upload (drag & drop)
- Real-time predictions
- Deep learning-based classification
- Interactive UI using Streamlit

---

## 🏗️ Project Structure

```
├── app.py
├── model/
├── requirements.txt
├── app_screenshot.jpg
└── README.md
```

---

## 🖥️ Installation & Setup

### Clone the repository
```
git clone https://github.com/trijesh61/Car-Damage-Detection-using-DeepLearning.git
cd Car-Damage-Detection-using-DeepLearning
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run the application
```
streamlit run app.py
```

---

## 🌍 Real-World Applications

- Insurance claim automation  
- Vehicle inspection systems  
- Fleet monitoring  
- Used car evaluation  

---

## ⚠️ Limitations

- No damage localization  
- Limited dataset  
- Works best on specific angles  

---

## 🚀 Future Improvements

- Severity prediction  
- Cost estimation  
