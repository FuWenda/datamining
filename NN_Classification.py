import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from preprocess import dataset

from sklearn.metrics import precision_score, recall_score, f1_score

from model import *

def predict(model, data_loader):
    predictions = []
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # 将输入移动到设备上（如果使用GPU）
            outputs = model(inputs)  # 模型前向传播
            _, predicted_labels = torch.max(outputs, dim=1)  # 获取预测标签
            predictions.extend(predicted_labels.tolist())
    return predictions
device = 'cuda' if torch.cuda.is_available() else 'cup'
print(device)

model=Net2().to(device)
# 定义 Loss 函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets=inputs.to(device),targets.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

# 定义测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets=inputs.to(device),targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy

# 假设你已经准备好了你的数据集：train_dataset、test_dataset

# 定义数据加载器


# 定义 5 折交叉验证
kfold = KFold(n_splits=5, shuffle=True,random_state=43)



acc_count = 0.0
precision_count = 0.0
recall_count = 0.0
f1_count = 0.0
num_folds = 5  # 设置折数

for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
    # 创建对应折的训练集和验证集的数据子集加载器
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

    # 创建新的模型并进行训练和测试
    model = model  # 请确保此处正确设置了您要使用的模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        train(model, train_loader, optimizer, criterion)

    accuracy = test(model, test_loader)
    predictions = predict(model, test_loader)

    # 计算精确度、召回率和 F1 分数
    targets = [label for _, label in test_subset]
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

average_accuracy = acc_count / num_folds
average_precision = precision_count / num_folds
average_recall = recall_count / num_folds
average_f1 = f1_count / num_folds

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1-Score: {average_f1:.4f}")