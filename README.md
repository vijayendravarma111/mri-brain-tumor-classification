# MRI Brain Tumor Classification System

A production-ready deep learning web application for multi-class brain tumor detection using MRI images.

This project implements an end-to-end pipeline — from model training using EfficientNet-B0 (PyTorch) to real-time image inference through a FastAPI-based web interface.

---

## Project Overview

Brain tumors such as Glioma, Meningioma, and Pituitary tumors require accurate imaging analysis for early detection and treatment planning.  

This system classifies MRI brain images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model is trained using transfer learning with EfficientNet-B0 and deployed as an interactive web application for real-time predictions.

---

##System Architecture

MRI Image Upload  
        ↓  
Preprocessing (Resize + Normalize)  
        ↓  
EfficientNet-B0 (Transfer Learning)  
        ↓  
Softmax Classification  
        ↓  
Prediction + Confidence Score + Report  

---

## Technology Stack

**Deep Learning**
- PyTorch
- Torchvision
- EfficientNet-B0 (ImageNet Pretrained)

**Backend**
- FastAPI
- Uvicorn

**Image Processing**
- PIL (Pillow)

**Frontend**
- TailwindCSS (embedded UI)

---

## Model Details

- Architecture: EfficientNet-B0
- Transfer Learning: ImageNet Pretrained Weights
- Image Size: 224x224
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Sampling Strategy: WeightedRandomSampler (class balancing)
- Data Augmentation:
  - Horizontal Flip
  - Small Rotation
- Test Accuracy: 83.31%  

---

## Project Structure

- brain_app.py # FastAPI web application
- infer.py # Standalone inference script
- models
  - best_model.pt # Trained model  
- requirements.txt
- README.md  


---

## ▶ How to Run Locally

### 1️⃣ Clone the repository

git clone https://github.com/vijayendravarma111/mri-brain-tumor-classification.git

cd mri-brain-tumor-classification

### 2️⃣ Install dependencies

pip install -r requirements.txt


### 3️⃣ Run the application

python brain_app.py


Upload an MRI image and get prediction instantly.

---

## Disclaimer

This system is developed for educational and research purposes only.

It is **not** a certified medical diagnostic tool and should not replace professional medical evaluation.

---

## Author

S.Vijayendra Varma  
B.Tech CSE (Data Science)  
Machine Learning & AI Enthusiast  

---

## Key Learning Outcomes

- Practical implementation of transfer learning
- Model fine-tuning using EfficientNet
- Handling class imbalance with weighted sampling
- End-to-end deployment of AI system
- Building production-ready FastAPI applications
- Designing user-friendly medical AI interfaces

---

## Future Improvements

- Grad-CAM visualization for explainability
- Docker containerization
- Cloud deployment (AWS / Render)
- Model versioning
- Performance benchmarking

---

If you found this project useful or interesting, feel free to star ⭐ the repository.


