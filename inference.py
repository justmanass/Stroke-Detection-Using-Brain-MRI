import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Load checkpoint
ckpt_path = r"D:\Kriti Model\outputs\densenet121_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2. Rebuild model
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3. Preprocessing
eval_tf = transforms.Compose([
    transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 4. Loop through test folders
y_true, y_pred = [], []
folders = {
    "normal": r"D:\Kriti Model\deep\test\normalMR",
    "stroke": r"D:\Kriti Model\deep\test\strokeMR"
}

for label_name, folder in folders.items():
    label = 0 if label_name == "normal" else 1
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            img = eval_tf(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(img)
                pred = torch.argmax(logits, dim=1).item()
            y_true.append(label)
            y_pred.append(pred)
            print(f"{fname} -> {pred}")

# 5. Metrics table
report = classification_report(y_true, y_pred, target_names=["Normal", "Stroke"], output_dict=True)
df = pd.DataFrame(report).transpose()
print("\nMetrics Table:")
print(df)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
