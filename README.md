# 🩺 PleuraMind — Chest X-Ray Disease Detection

AI-powered chest X-ray disease detection system using **EfficientNetV2-M**, deep learning, and **Grad-CAM explainability**.

PleuraMind predicts **multiple thoracic diseases from a single chest X-ray image** and provides visual explanations of the model's decision.

---

# 🚀 Live Demo

Try the deployed model:

👉 https://huggingface.co/spaces/vedgharat/pleuramind

## 🤖 Model Weights

The trained model weights are hosted on Hugging Face Model Hub.

👉 https://huggingface.co/vedgharat/pleuramind-model

The application automatically downloads the model during inference using `hf_hub_download`.

---

# 📌 Overview

Chest X-ray interpretation is a critical task in medical diagnostics.  
This project builds a **multi-label deep learning model** capable of predicting **8 thoracic conditions simultaneously**.

The system analyzes an X-ray image and outputs probabilities for each disease.

### Predicted Diseases

| Disease |
|------|
| Atelectasis |
| Cardiomegaly |
| Consolidation |
| Edema |
| Pleural Effusion |
| Pneumonia |
| Pneumothorax |
| No Finding |

---

# 🧠 Model Architecture

The model uses **transfer learning** from EfficientNetV2-M and a custom classification head.

### Architecture Components

- Backbone: **EfficientNetV2-M**
- Pooling: **GeM (Generalized Mean Pooling)**
- Custom classification head
- Multi-label sigmoid output layer

---

# ⚙️ Training Configuration

| Parameter | Value |
|------|------|
| Image Size | 320 × 320 |
| Batch Size | 32 |
| Epochs | 8 |
| Optimizer | AdamW |
| Scheduler | Cosine Warmup |
| Mixed Precision | Yes |
| Loss Function | Masked BCEWithLogits |
| Dataset | CheXpert |

### Training Techniques Used

- Transfer learning
- Mixed precision training
- Class imbalance handling
- Uncertain label handling
- Backbone freezing during warmup
- Fine-tuning after warmup
- Test-Time Augmentation (TTA)

---

# 📊 Model Performance

### Best Validation Results

| Disease | AUC |
|------|------|
| Atelectasis | 0.7857 |
| Cardiomegaly | 0.8282 |
| Consolidation | 0.9355 |
| Edema | 0.9152 |
| Pleural Effusion | 0.9370 |
| Pneumonia | 0.8319 |
| Pneumothorax | 0.7884 |
| No Finding | 0.9041 |

---

# 📈 Training Curves

![Training Curves](results/training_curves.png)

---

# 📉 ROC Curves

![ROC Curves](results/roc_curves.png)

---

# 🔬 Model Explainability (Grad-CAM)

Grad-CAM highlights the regions of the X-ray that influenced the model’s predictions.

| Disease | Visualization |
|------|------|
| Atelectasis | ![](results/gradcam_Atelectasis.png) |
| Cardiomegaly | ![](results/gradcam_Cardiomegaly.png) |
| Consolidation | ![](results/gradcam_Consolidation.png) |
| Edema | ![](results/gradcam_Edema.png) |
| Pleural Effusion | ![](results/gradcam_Pleural_Effusion.png) |
| Pneumonia | ![](results/gradcam_Pneumonia.png) |
| Pneumothorax | ![](results/gradcam_Pneumothorax.png) |
| No Finding | ![](results/gradcam_No_Finding.png) |

---

# 📂 Project Structure

```
PleuraMind
│
├── app.py
├── inference.py
├── model_definition.py
├── requirements.txt
│
├── results
│   ├── training_curves.png
│   ├── roc_curves.png
│   ├── gradcam_*.png
│
├── notebook
│   └── training_notebook.ipynb
│
└── README.md
```

---

# 🚀 Running Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/vedgharat/Pleura-Mind.git
cd Pleura-Mind
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

The interface will open at:

```
http://127.0.0.1:7860
```

---

# 🧪 Example Inference

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model_definition import CheXpertModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CheXpertModel(num_classes=8)

checkpoint = torch.load("model/best_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

model.to(DEVICE)
model.eval()

transform = T.Compose([
    T.Resize((320,320)),
    T.ToTensor()
])

img = Image.open("xray.jpg").convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(img)
    probs = torch.sigmoid(logits)

print(probs)
```

---

# 📁 Output Files

### validation_predictions.csv

Contains predicted probabilities and binary predictions for each disease.

### model_summary.csv

Contains evaluation metrics and training statistics.

---

# ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.

It is **not a medical diagnostic tool** and must not be used for clinical decision-making.

Always consult qualified medical professionals for medical diagnosis.

---

# 📚 Dataset

The model is trained using the **CheXpert Dataset** from Stanford University.

https://stanfordmlgroup.github.io/competitions/chexpert/

---

# 👨‍💻 Author

**Ved Gharat**

Computer Engineering Student  
Interested in **Deep Learning, AI Systems, and Computer Vision**

GitHub: https://github.com/vedgharat
