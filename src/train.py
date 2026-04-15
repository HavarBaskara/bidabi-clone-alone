from pathlib import Path
import json
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


DATA_DIR = Path("data/raw/images")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

BATCH_SIZE = 16
EPOCHS = 5
SEED = 42
IMG_SIZE = 224


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_splits(n: int, train_ratio=0.7, val_ratio=0.15):
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def build_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_dataset = datasets.ImageFolder(DATA_DIR)
    train_idx, val_idx, test_idx = make_splits(len(base_dataset))

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
    test_dataset = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return base_dataset.class_to_idx, train_loader, val_loader, test_loader


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = get_device()

    class_to_idx, train_loader, val_loader, test_loader = build_loaders()
    num_classes = len(class_to_idx)

    if num_classes < 2:
        raise ValueError("Il faut au moins 2 classes pour entraîner le modèle.")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "img_size": IMG_SIZE,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                },
                BEST_MODEL_PATH,
            )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    metadata = {
        "device": str(device),
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "epochs": EPOCHS,
        "best_val_acc": round(best_val_acc, 4),
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),
        "history": history,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMeilleur modèle sauvegardé : {BEST_MODEL_PATH}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Métadonnées sauvegardées : {METADATA_PATH}")


if __name__ == "__main__":
    train()