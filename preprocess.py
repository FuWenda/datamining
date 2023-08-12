import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split
from DataCleaning import df_new

class mydataset(Dataset):
    def __init__(self,data_df):
        data=torch.from_numpy(data_df.values)
        self.feature=data[:,:-1]
        self.labels=data[:,-1].to(int)-1
        # print(self.labels)
        # exit(0)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        x=self.feature[idx]
        y=self.labels[idx]
        return x,y

dataset=mydataset(df_new)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(66)
trainset, testset = random_split(dataset, [train_size, test_size])

trainloader=DataLoader(dataset=trainset,batch_size=32,shuffle=True)
testloader=DataLoader(dataset=testset,batch_size=32,shuffle=True)

# 打印训练集和验证集的大小
print("训练集大小：", len(trainset))
print("验证集大小：", len(testset))




