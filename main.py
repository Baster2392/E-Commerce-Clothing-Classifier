import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall

from torchvision import datasets
import torchvision.transforms as transforms
from clothing_classifier_model import ClothingClassifierModel


# Create dataloaders
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

model = ClothingClassifierModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
# Train loop
model.train()
for epoch in range(epochs):
    print(f"Running epoch {epoch + 1}.")
    total_loss = 0.0
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1} ended with total loss {total_loss}.")

# Evaluate model
print(f"Finished training for {epochs} epochs.")
model.eval()
num_classes = 10
metric = Accuracy(task="multiclass", num_classes=num_classes)
precision = Precision(task="multiclass", num_classes=num_classes, average=None)
recall = Recall(task="multiclass", num_classes=num_classes, average=None)

test_loss = 0.0
predictions = []
with torch.no_grad():
    for idx, (data, target) in enumerate(test_loader):
        output = model(data)
        predictions.extend(output.argmax(dim=-1).tolist())
        test_loss += criterion(output, target).item()
        metric(output.argmax(dim=-1), target)
        precision(output.argmax(dim=-1), target)
        recall(output.argmax(dim=-1), target)

    accuracy = float(metric.compute())
    precisions = precision.compute()
    recalls = recall.compute()
    print("------------RESULTS------------")
    print(f"Accuracy on test set: {accuracy}")
    print("Precision and recall:")
    for idx in range(num_classes):
        print(f"Label: {idx}, precision {precisions[idx]}, recall {recalls[idx]}")
