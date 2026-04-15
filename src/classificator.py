"""
Training pipeline for ResNet‑18 full fine‑tuning with MixUp, t‑SNE, UMAP,
and extended evaluation metrics (confusion matrix, ROC, hardest samples).
"""

# --- Importations ---
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import seaborn as sns

# --- UMAP (optionnel) ---
try:
    import umap
    umap_available = True
except ImportError:
    print("UMAP not installed — skipping UMAP visualization.")
    umap_available = False


# --- Seed pour reproductibilité ---
def set_seed(seed=42):
    """
    Fixes all relevant random seeds to ensure reproducible training.

    Parameters
    ----------
    seed : int, optional
        Random seed value used for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# --- Constantes globales ---
H = 256
W = 256
BATCH_SIZE = 32
DATA_DIR = "./Flipkart/Sorted/"
NUM_EPOCHS = 20
PATIENCE = 3

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Chargement du dataset ---
dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

NUM_CLASSES = len(dataset.classes)
print("Catégories détectées :", dataset.classes)

# --- Split train/val/test ---
total_len = len(dataset)
train_size = int(0.6 * total_len)
val_size = int(0.2 * total_len)
test_size = total_len - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Train: {len(train_dataset)}, "
      f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# --- Modèle ResNet18 ---
def create_resnet18(num_classes):
    """
    Creates a ResNet‑18 model with full fine‑tuning and a custom classifier head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.

    Returns
    -------
    torch.nn.Module
        Modified ResNet‑18 model.
    """
    model = resnet18(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


# --- Device ---
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print("Utilisation de l'appareil:", device)

model = create_resnet18(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# --- MixUp ---
def mixup_data(x, y, alpha=0.4):
    """
    Applies MixUp augmentation to a batch of images and labels.

    Parameters
    ----------
    x : torch.Tensor
        Batch of input images.
    y : torch.Tensor
        Batch of labels.
    alpha : float, optional
        Beta distribution parameter controlling MixUp intensity.

    Returns
    -------
    mixed_x : torch.Tensor
        Mixed images.
    y_a : torch.Tensor
        Original labels.
    y_b : torch.Tensor
        Shuffled labels.
    lam : float
        MixUp interpolation coefficient.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# --- Suivi des métriques ---
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float("inf")
best_val_acc = 0.0
patience_counter = 0


# --- Boucle d'entraînement ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=0.4
        )

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = (
            lam * criterion(outputs, targets_a)
            + (1 - lam) * criterion(outputs, targets_b)
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} — "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "best_model_resnet18_finetuned.pth"
        )
        print("→ Nouveau meilleur modèle sauvegardé")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    scheduler.step()

print("Entraînement terminé. Meilleure Val Acc:", best_val_acc)

# --- Graphiques Loss & Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (ResNet18 full FT + MixUp)")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy (ResNet18 full FT + MixUp)")
plt.legend()
plt.show()


# --- Évaluation sur le test ---
model.load_state_dict(torch.load("best_model_resnet18_finetuned.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []


def evaluate_model(model, loader):
    """
    Runs inference on a dataloader and collects predictions, labels and probabilities.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test or validation dataloader.

    Returns
    -------
    preds : np.ndarray
        Predicted class indices.
    labels : np.ndarray
        Ground‑truth labels.
    probs : np.ndarray
        Softmax probabilities for each class.
    """
    preds = []
    labels = []
    probs = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            p = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return (
        np.array(preds),
        np.array(labels),
        np.array(probs)
    )


all_preds, all_labels, all_probs = evaluate_model(model, test_loader)

# --- Rapport de classification ---
print(classification_report(all_labels, all_preds, target_names=dataset.classes))


# --- Confusion Matrix ---
def plot_confusion_matrix(cm, classes):
    """
    Displays a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    classes : list of str
        Class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (ResNet18 full FT + MixUp)")
    plt.show()


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, dataset.classes)


# --- Per-class accuracy ---
def compute_per_class_accuracy(model, loader, num_classes):
    """
    Computes accuracy for each class separately.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test dataloader.
    num_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Per-class accuracy values.
    """
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            matches = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i].item()
                correct[label] += matches[i].item()
                total[label] += 1

    return correct / total


per_class_acc = compute_per_class_accuracy(
    model, test_loader, NUM_CLASSES
)

plt.figure(figsize=(10, 5))
plt.bar(dataset.classes, per_class_acc)
plt.ylabel("Accuracy")
plt.title("Per-class Accuracy (ResNet18 full FT + MixUp)")
plt.xticks(rotation=45)
plt.show()


# --- ROC curves (One-vs-Rest) ---
def plot_roc_curves(labels, probs, classes):
    """
    Plots ROC curves for each class (one-vs-rest).

    Parameters
    ----------
    labels : np.ndarray
        Ground‑truth labels.
    probs : np.ndarray
        Softmax probabilities.
    classes : list of str
        Class names.
    """
    y_bin = label_binarize(labels, classes=list(range(len(classes))))

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (ResNet18 full FT + MixUp)")
    plt.legend()
    plt.show()


plot_roc_curves(all_labels, all_probs, dataset.classes)


# --- Hardest samples ---
def compute_hardest_samples(model, loader, classes, top_k=12):
    """
    Identifies the samples with the highest loss.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    loader : DataLoader
        Test dataloader.
    classes : list of str
        Class names.
    top_k : int, optional
        Number of hardest samples to display.

    Returns
    -------
    None
    """
    criterion_nr = nn.CrossEntropyLoss(reduction="none")

    losses = []
    imgs = []
    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            y = y.to(device)

            outputs = model(images)
            batch_losses = criterion_nr(outputs, y)

            losses.extend(batch_losses.cpu().numpy())
            imgs.extend(images.cpu())
            labels.extend(y.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())

    losses = np.array(losses)
    idx_sorted = np.argsort(losses)[::-1]
    top_k = min(top_k, len(idx_sorted))

    plt.figure(figsize=(12, 10))
    for i in range(top_k):
        idx = idx_sorted[i]
        img = imgs[idx].permute(1, 2, 0).numpy()
        img = (
            img * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        ).clip(0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(
            f"Loss={losses[idx]:.2f}\n"
            f"True={classes[labels[idx]]}\n"
            f"Pred={classes[preds[idx]]}"
        )
        plt.axis("off")

    plt.suptitle("Top Hardest Samples (Highest Loss)")
    plt.show()


compute_hardest_samples(model, test_loader, dataset.classes)


# --- Embeddings t-SNE & UMAP ---
def extract_features(model, x):
    """
    Extracts convolutional features from ResNet‑18 before the classifier.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ResNet‑18.
    x : torch.Tensor
        Input batch.

    Returns
    -------
    torch.Tensor
        Flattened feature vectors.
    """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    return torch.flatten(x, 1)


embeddings = []
labels_list = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        feats = extract_features(model, images)

        embeddings.append(feats.cpu().numpy())
        labels_list.extend(labels.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels_list = np.array(labels_list)


# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
tsne_emb = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, cls in enumerate(dataset.classes):
    idx = labels_list == i
    plt.scatter(tsne_emb[idx, 0], tsne_emb[idx, 1], label=cls, alpha=0.6)

plt.legend()
plt.title("t-SNE Embedding Visualization")
plt.show()


# --- UMAP ---
if umap_available:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    umap_emb = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(dataset.classes):
        idx = labels_list == i
        plt.scatter(umap_emb[idx, 0], umap_emb[idx, 1], label=cls, alpha=0.6)

    plt.legend()
    plt.title("UMAP Embedding Visualization")
    plt.show()
