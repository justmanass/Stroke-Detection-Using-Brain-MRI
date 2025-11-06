import os
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# Config
# -----------------------------
CLASS_NAMES = ["normal", "stroke"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class MRImageDataset(Dataset):
    def __init__(self, file_label_list, transform=None):
        self.samples = file_label_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def gather_files_from_dirs(dir_normal, dir_stroke):
    def list_images(root):
        files = []
        root = Path(root)
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(str(p))
        return files

    normal_files = list_images(dir_normal)
    stroke_files = list_images(dir_stroke)

    normal_labeled = [(f, 0) for f in normal_files]
    stroke_labeled = [(f, 1) for f in stroke_files]
    return normal_labeled + stroke_labeled


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_tf, eval_tf


# -----------------------------
# Models
# -----------------------------
def build_model(model_name="resnet50", num_classes=2):
    if model_name.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    elif model_name.lower() == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError("model_name must be 'resnet50' or 'densenet121'")


# -----------------------------
# Metrics
# -----------------------------
def binary_metrics_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return acc, precision, recall, f1, tp, tn, fp, fn


# -----------------------------
# Training / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc, precision, recall, f1, tp, tn, fp, fn = binary_metrics_from_logits(all_logits, all_labels)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def compute_class_weights(samples):
    counts = [0, 0]
    for _, y in samples:
        counts[y] += 1
    total = sum(counts)
    weights = [total / max(1, c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "densenet121"])
    parser.add_argument("--train_normal_dir", type=str, default=r"D:\Kriti Model\deep\train\2- Control\NormalMR")
    parser.add_argument("--train_stroke_dir", type=str, default=r"D:\Kriti Model\deep\train\1- Stroke\StrokeMR")
    parser.add_argument("--test_normal_dir", type=str, default=r"D:\Kriti Model\deep\test\normalMR")
    parser.add_argument("--test_stroke_dir", type=str, default=r"D:\Kriti Model\deep\test\strokeMR")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_samples = gather_files_from_dirs(args.train_normal_dir, args.train_stroke_dir)
    test_samples = gather_files_from_dirs(args.test_normal_dir, args.test_stroke_dir)

    random.shuffle(train_samples)
    n_val = int(len(train_samples) * args.val_split)
    val_samples = train_samples[:n_val]
    train_samples = train_samples[n_val:]

    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = MRImageDataset(train_samples, transform=train_tf)
    val_ds = MRImageDataset(val_samples, transform=eval_tf)
    test_ds = MRImageDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(args.model_name, num_classes=2).to(device)

    # Freeze backbone for a few epochs, then unfreeze
    for p in model.parameters():
        p.requires_grad = True  # fine-tune all by default
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_path = os.path.join(args.out_dir, f"{args.model_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f} | "
              f"val_precision={val_metrics['precision']:.4f} | val_recall={val_metrics['recall']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model_name,
                "img_size": args.img_size,
                "class_names": CLASS_NAMES,
            }, best_path)
            print(f"Saved best model to {best_path}")

    # Test evaluation with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["model_name"], num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nTest results:")
    print(f"loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.4f} | "
          f"precision={test_metrics['precision']:.4f} | recall={test_metrics['recall']:.4f} | "
          f"f1={test_metrics['f1']:.4f}")
    print(f"Confusion Matrix: TP={test_metrics['tp']} TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']}")


if __name__ == "__main__":
    main()
