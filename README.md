# MRI-Brain-Tumor-Detection
CNN Project
# 🧠 Lightweight Brain Tumor Detection Using CNNs

## 📌 Project Overview
Early brain tumor detection is critical for effective clinical intervention. MRI is one of the most reliable imaging modalities for tumor diagnosis, and deep learning offers the potential to automate and enhance this process.

This project focuses on **building an efficient, lightweight deep learning system** for brain tumor detection and classification from MRI scans. The core objective is to **balance accuracy with computational efficiency**, making the solution suitable for **deployment on limited hardware**.

📄 *Detailed methodology and results are documented in the project report* : https://drive.google.com/file/d/1Vg94Sz6T6IiJvc51nJFhrNX9ymlTVXdq/view?usp=sharing

---

## 🎯 Project Objectives
- Develop a **custom lightweight CNN** for MRI-based brain tumor classification
- Compare its performance against a **pre-trained ResNet50 model**
- Analyze trade-offs between **model complexity, accuracy, and deployability**
- Demonstrate real-time prediction via a **Gradio web interface**

---

## 🗂️ Dataset
- **Source:** Kaggle Brain Tumor MRI Dataset : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data
- **Classes:**
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

### Files:
- [A] EDA_Brain_Tumor_Detection_Omkar.ipynb - Performed and carried out all the Data Preprocessing and EDA steps
- [B] Model_Brain_Tumor_Detection_Omkar.ipynb - Trained the Model from scratch
- [C] Custom CNN Model (.h5 file) : https://drive.google.com/file/d/1m_bjUGG84mQhAyTrXZuFVjufv2bRMWCi/view?usp=sharing
- [D] ResNet50 Model (.h5 file): https://drive.google.com/file/d/1aED7_O9J9_xziqq9DGMA2ihYeZGfJF8u/view?usp=sharing
- [E] Report_Brain_Tumor_Detection.pdf
- [F] Presentation_Brain_Tumor_Detection.pdf

### Data Preparation Steps
- Folder-based image structuring
- Image integrity checks (corrupted file removal)
- Duplicate image removal using hashing
- **Class balancing via under-sampling** to avoid model bias

Balanced data ensured fair training and evaluation across all classes.

---

## 🧠 Models Implemented

### 1️⃣ Custom Lightweight CNN (Built from Scratch)
Designed specifically for efficiency and clinical feasibility.

**Architecture Highlights:**
- Convolutional layers with ReLU activation
- MaxPooling for spatial reduction
- Dense layers for classification
- Dropout for overfitting control
- Softmax output layer for multi-class prediction

**Key Characteristics:**
- ~198K trainable parameters
- Fast training and inference
- Low memory footprint
- High interpretability

---

### 2️⃣ ResNet50 (Transfer Learning)
A deep residual network pre-trained on ImageNet.

**Approach:**
- Frozen base ResNet50 layers
- Custom classification head:
  - Global Average Pooling
  - Dense + Dropout layers
  - Softmax output

**Key Characteristics:**
- ~25.6 million parameters
- High accuracy
- Higher computational cost
- Increased risk of overfitting on small datasets

---

## ⚖️ Model Comparison

| Feature | Custom CNN | ResNet50 |
|------|----------|----------|
| Total Parameters | ~198K | ~25.6M |
| Training Speed | Very Fast | Slower |
| GPU Memory | Low | High |
| Overfitting Risk | Low | Higher |
| Deployment Suitability | Excellent | Limited |
| Clinical Feasibility | High | Moderate |

---

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall |
|-----|---------|----------|--------|
| Custom CNN | 91% | 92% | 91% |
| ResNet50 | **93%** | **94%** | **94%** |

- ResNet50 achieved slightly higher accuracy
- Custom CNN delivered **competitive performance with drastically fewer parameters**

Confusion matrices showed strong diagonal dominance for both models, indicating reliable class predictions.

---

## 📈 Training Analysis
- Custom CNN demonstrated **stable learning curves**
- Validation accuracy closely tracked training accuracy
- ResNet50 converged faster but required careful monitoring to avoid overfitting

---

## 🖥️ Deployment & Demo
Both models were deployed using a **Gradio web interface**, allowing:
- MRI image upload
- Real-time tumor classification
- Probability visualization for all classes

This validates the system’s readiness for practical usage.

---

## ✅ Final Verdict
- **ResNet50** is ideal when maximum accuracy is required and resources are abundant
- **Custom Lightweight CNN** is the better choice for:
  - Limited hardware environments
  - Faster inference
  - Clinical and real-time applications

The project demonstrates that **well-designed lightweight models can rival deep pre-trained networks** while being far more deployable.

---

## 🚀 Future Scope
- Integration with hospital PACS systems
- Model quantization for edge deployment
- Validation on larger multi-center MRI datasets
- Extension to tumor segmentation tasks

---

## 👨‍💻 Authors
- **Omkar Umesh Gadade**
---

