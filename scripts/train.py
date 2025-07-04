import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import FrameSequenceDataset, get_transform
from model import ConvLSTMClassifier
from tqdm import tqdm
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Hyperparameters
batch_size = 4
epochs = 10
lr = 1e-4
model_save_path = "best_model.pth"

# Load datasets
train_set = FrameSequenceDataset("processed_data", split="train", transform=get_transform())
val_set = FrameSequenceDataset("processed_data", split="val", transform=get_transform())

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# Model
model = ConvLSTMClassifier().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        clips, labels = clips.to(device), labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"\nTrain Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Validation Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), model_save_path)
        best_val_acc = val_acc
        print(f"✅ Best model saved (val_acc={val_acc:.4f})")

