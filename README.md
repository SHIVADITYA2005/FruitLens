FruitLens: Random Forest Image Classifier

FruitLens is an interactive machine learning application that identifies and categorizes fruit species using a custom-trained ensemble learning model. Built with Scikit-Learn and OpenCV, this project demonstrates a complete AI lifecycle—from data preprocessing and model training to real-time inference via a web-based interface.

🧠 Model Architecture

The model is trained on a custom dataset of fruit images and utilizes a Random Forest algorithm structure:

Input Layer: Resizes raw RGB images to a $64 \times 64$ pixel grid and flattens them into 12,288-dimensional arrays (64x64x3).

Classifier: An ensemble of 100 decision trees (n_estimators=100) utilizing the Random Forest algorithm for robust, non-linear feature extraction and classification.

Output Layer: Outputs the predicted fruit category based on majority voting across 5 possible classes (Apple, Banana, Mango, Strawberry, Grape).

🛠️ Tech Stack

Machine Learning Framework: Scikit-Learn

Web UI / Deployment: Gradio (Interactive Image Upload)

Image Processing: OpenCV (cv2), NumPy

🚀 Getting Started

Prerequisites

Ensure you have Python installed, then install the required dependencies:

pip install scikit-learn gradio numpy opencv-python joblib


Running the Application

Clone the repository and navigate to the project directory.

Run the Python training script to process the images, train the model, and save the serialized weights:

python prp.py


Usage: Run the inference script or the finalPRP.ipynb notebook to launch the Gradio web interface. Once the terminal provides a local URL (e.g., http://127.0.0.1:7860), open it in your browser. Upload any image of the supported fruits, and the model will display the prediction in real-time.

⚙️ Data Pipeline & Preprocessing

To ensure high accuracy on user-uploaded images, the data pipeline automatically preprocesses the inputs to match the training conditions:

Reads the raw images using OpenCV.

Resizes the images to a standardized $64 \times 64$ pixel grid to maintain uniform feature extraction.

Flattens the 3-channel (RGB) 2D image into a 1D numerical array.

Handles dataset generation by dynamically shuffling and capping training samples at a maximum of 800 images per fruit category to prevent severe class imbalance.

📊 Training Results

The model is trained using the RandomForestClassifier with a fixed random state (random_state=42) for reproducibility. By leveraging ensemble learning, it efficiently navigates high-dimensional pixel data without suffering from overfitting, achieving robust classification accuracy across the targeted fruit categories.

📜 License

Distributed under the MIT License.
