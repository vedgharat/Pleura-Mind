import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from model_definition import CheXpertModel  # import your model class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISEASES = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
            "Pleural Effusion","Pneumonia","Pneumothorax","No Finding"]

# load model
ckpt = torch.load("model/best_model.pt", map_location=DEVICE)
model = CheXpertModel(num_classes=len(DISEASES), dropout=0.3, pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()

transform = T.Compose([
    T.Resize((320,320)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return dict(zip(DISEASES, probs))

if __name__ == "__main__":
    preds = predict("results/sample_xray.jpg")
    for k,v in preds.items():
        print(f"{k}: {v:.3f}")