import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from dataset import RAVDESSDataset
from models.seformer import SEFormer
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = RAVDESSDataset("data/raw_audio")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Model
model = SEFormer().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

scaler = GradScaler()

epochs = 5
best_acc = 0

train_losses = []
val_accuracies = []

start_time = time.time()

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if device.type == "cuda":
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Training Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    val_accuracies.append(accuracy)

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")

    # Save best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_seformer.pth")
        print("Best Model Saved!")

end_time = time.time()
print(f"\nTotal Training Time: {(end_time - start_time)/60:.2f} minutes")

# Plot Loss
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")

# Plot Accuracy
plt.figure()
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("validation_accuracy.png")

print("Training Complete!")