import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess import dataset
from model import *



def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = Net2().to(device)
Epoch=30#

# Define Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define train and test functions
def train(model, train_loader, optimizer, criterion, epoch, writer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Write training loss to TensorBoard
        writer.add_scalar('Train Loss', loss.item(), epoch * len(train_loader) + batch_idx)


def test(model, test_loader, criterion, epoch, writer):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    average_loss = loss_total / len(test_loader)

    writer.add_scalar('Val Loss', average_loss, epoch)

    return accuracy


kfold = KFold(n_splits=5, shuffle=True, random_state=43)

acc_count = 0.0
precision_count = 0.0
recall_count = 0.0
f1_count = 0.0
num_folds = 5  # Set the number of folds

for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
    # Create data loaders for the corresponding train and validation subsets
    outdir=f"./runs/flod {fold}"
    writer = SummaryWriter(outdir)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, test_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

    # Create a new model and optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(Epoch):
        train(model, train_loader, optimizer, criterion, epoch, writer)
        test(model, val_loader, criterion, epoch, writer)

    accuracy = test(model, val_loader, criterion, epoch, writer)

    # Skip if accuracy is None
    if accuracy is None:
        continue

    predictions = predict(model, val_loader)

    # Calculate precision, recall, and F1-score
    targets = [label for _, label in val_subset]
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    acc_count += accuracy
    precision_count += precision
    recall_count += recall
    f1_count += f1
    print(f"Fold {fold + 1}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    writer.close()



average_accuracy = acc_count / num_folds
average_precision = precision_count / num_folds
average_recall = recall_count / num_folds
average_f1 = f1_count / num_folds

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1-Score: {average_f1:.4f}")


#tensorboard --logdir="D:\mywork\dataMining\runs"