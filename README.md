# 🧠 Brain Tumor MRI Classification

A deep learning web application that classifies brain MRI scans into four categories using ANN, CNN, and Transfer Learning (MobileNetV2), served via a FastAPI backend.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview

This project was developed as a college neural network assignment. Given a brain MRI image, the application predicts which of the following tumor types is present:

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `notumor` | No tumor detected |
| `pituitary` | Pituitary tumor |

Three architectures were trained and compared:
- **ANN** — Fully connected network on flattened image vectors
- **CNN** — Custom convolutional network (3 Conv blocks + Dense head)
- **Transfer Learning** — MobileNetV2 pretrained on ImageNet with a custom classification head

The final deployed model is the **Transfer Learning (MobileNetV2)** model saved as `brain_tumor_model.keras`.

---

## 🖥️ Demo

> Add screenshots of the running app below:

| Upload Screen | Prediction Result |
|:---:|:---:|
| ![Upload](https://raw.githubusercontent.com/Mohanad06/Brain-Tumor/main/screenshots/Te-no_3.jpg) | ![Result](https://raw.githubusercontent.com/Mohanad06/Brain-Tumor/main/screenshots/Result.PNG) |

> Sample predictions from the deployed web app

---

## 📂 Dataset

The model was trained on the **Brain Tumor MRI Dataset** from Kaggle.

| Split | Samples |
|-------|---------|
| Training | ~5,712 |
| Validation | ~20% of training |
| Testing | ~1,311 |

> 📥 **Dataset Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset]

---

## 🤖 Models

| Model | Architecture | Input |
|-------|-------------|-------|
| ANN | Dense(512→256→128→4) + Dropout | Flattened 224×224×3 vector |
| CNN | 3× (Conv2D + BN + MaxPool) → Dense(256→4) | 224×224×3 image |
| Transfer Learning | MobileNetV2 (frozen) → Dense(128→64→32→16→4) | 224×224×3 image |

All models use:
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping + ReduceLROnPlateau

> 💾 **Trained Model (.keras):** [https://drive.google.com/drive/folders/1BKrexher9XpBBmXFHt-h6KgJV4-wGPNt?usp=sharing]

---

## 📁 Project Structure

```
brain-tumor-classifier/
│
├── app.py                  # FastAPI application
├── requirements.txt        # Python dependencies
├── Run.bat                 # One-click launcher (Windows)
├── brain_tumor_model.keras # Saved trained model  ← download from Drive link above
│
├── Brain_Tumor.ipynb       # Training notebook (ANN + CNN + Transfer Learning)
│
├── templates/
│   └── index.html          # Frontend UI
│
└── screenshots/            # App screenshots (for README)
```

---

## ▶️ How to Run

### Prerequisites
- Python 3.9 or higher — [Download here](https://www.python.org/downloads/)
- Make sure to tick **"Add Python to PATH"** during installation

---

### ⚡ Option 1 — One Click (Windows)

1. Download or clone this repository
2. Download `brain_tumor_model.keras` from the [Google Drive link above](https://drive.google.com/drive/folders/1BKrexher9XpBBmXFHt-h6KgJV4-wGPNt?usp=sharing) and place it in the project root folder
3. Double-click **`Run.bat`**
4. The app will install dependencies automatically and open in your browser

---

### 🛠️ Option 2 — Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/Mohanad06/Brain-Tumor.git
cd brain-tumor-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the model from Drive and place it in the project folder
#    (see the Model link in the Models section above)

# 4. Run the app
python app.py
```

Then open your browser and go to: **http://localhost:8000**

---

### 🔮 Using the App

1. Open the app in your browser
2. Upload a brain MRI image (`.jpg` or `.png`)
3. Click **Predict**
4. The model will return the predicted tumor class with confidence score

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras |
| Pretrained Model | MobileNetV2 (ImageNet) |
| Backend API | FastAPI |
| Data Augmentation | Keras ImageDataGenerator |
| Evaluation | scikit-learn (classification report, confusion matrix) |
| Frontend | HTML / CSS / JavaScript |

---
## 📊 Results

| Model | Accuracy |
|-------|----------|
| ANN | 44.81% |
| CNN | 81.31% |
| MobileNetV2 | 91.31% (Best) |

---
## Future Improvements

- Deploy on cloud (Render / Hugging Face / AWS)
- Add Grad-CAM visualization for explainability
- Improve model accuracy with fine-tuning

---
## 👤 Author

- **Name:** Mohanad Mostafa  
- **College:** BFCAI  
- **Course:** Neural Networks

---

## 📄 License

This project is for educational purposes only.
