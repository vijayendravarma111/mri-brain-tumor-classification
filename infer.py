import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, torch, json
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn

CKPT = "models/best_model.pt"

def load():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(ckpt["class_names"]))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt["class_names"], ckpt["img_size"]

def predict(img_path):
    m, names, sz = load()
    tfm = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = m(img)
        prob = torch.softmax(logits,1).squeeze()
        idx = int(prob.argmax().item())
        return names[idx], float(prob[idx].item())

def render_report(label, conf):
    txt = f"""AI Report (Brain MRI)
Impression: Probable {label} (confidence {conf:.2f})
Details: Model found features consistent with {label}. 
Recommendation: This is NOT a medical diagnosis. Consult a radiologist/physician for confirmation."""
    return txt

if __name__=="__main__":
    path = sys.argv[1]
    lab, conf = predict(path)
    print(render_report(lab, conf))
