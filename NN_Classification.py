import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from proprecess import dataset



def predict(model, data_loader):
    predictions = []
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for inputs, _ in data_loader:
            # inputs = inputs.to(device)  # 将输入移动到设备上（如果使用GPU）
            outputs = model(inputs)  # 模型前向传播
            _, predicted_labels = torch.max(outputs, dim=1)  # 获取预测标签
            predictions.extend(predicted_labels.tolist())
    return predictions
class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()

        self.work = nn.Sequential(
            nn.Linear(278, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16))
    def forward(self, x):
        out = self.work(x)
        return out


class Net2(nn.Module):
    def __init__(self, input_size=278, output_size=16):
        super(Net2, self).__init__()
        self.tree = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        x = self.tree(x)
        return x

# 假设你已经定义好了你的神经网络模型：model

model=Net1()
# 定义 Loss 函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    for inputs, targets in train_loader:
        targets=targets
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
            targets = targets
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

#python
from sklearn.metrics import precision_score, recall_score, f1_score

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

    for epoch in range(30):  # 30个 epochs，可以根据需要进行修改
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