# FruitLens: AI-Powered Fruit Classification

**FruitLens** is an end-to-end computer vision pipeline designed to identify and categorize fruit species from raw image data. Built using Python, this project demonstrates the practical application of ensemble learning in image recognition, providing a seamless transition from model training to real-time web deployment.

## 🍎 Supported Categories
The model is currently trained to recognize five distinct fruit types:
* **Apple**, **Banana**, **Mango**, **Strawberry**, and **Grape**.

## 🛠️ Technical Architecture
* **Image Preprocessing:** Utilizes **OpenCV** to normalize datasets, resizing raw images to a uniform $64 \times 64$ pixel grid and flattening them into numerical feature vectors.
* **Ensemble Learning:** Employs a **Random Forest Classifier** with 100 estimators to navigate high-dimensional pixel data and achieve robust classification.
* **Interactive Deployment:** Integrated with **Gradio** to provide a user-friendly web interface for instant drag-and-drop image classification.
* **Model Persistence:** Uses **joblib** for efficient serialization, ensuring the trained weights (`fruit_classifier.pkl`) are portable and production-ready.



[Image of a machine learning workflow for image classification]


## 📁 Project Structure
* `prp.py`: The training script responsible for data loading, preprocessing, and model fitting.
* `finalprp.py`: The inference script that loads the saved model and launches the Gradio UI.
* `fruit_classifier.pkl`: The serialized Random Forest model (~14.7 MB).

## 🚀 Getting Started

### Prerequisites
Install the required dependencies using pip:
```bash
pip install opencv-python numpy scikit-learn gradio joblib
