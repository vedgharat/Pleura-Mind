import torch
from PIL import Image
import torchvision.transforms as T
from model_definition import CheXpertModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DISEASES = [
"Atelectasis",
"Cardiomegaly",
"Consolidation",
"Edema",
"Pleural Effusion",
"Pneumonia",
"Pneumothorax",
"No Finding"
]

model = CheXpertModel(num_classes=8)

ckpt = torch.load("model/best_model.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

model.to(DEVICE)
model.eval()

transform = T.Compose([
    T.Resize((320,320)),
    T.ToTensor()
])

def predict(image):

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    return {d: float(p) for d,p in zip(DISEASES, probs)}