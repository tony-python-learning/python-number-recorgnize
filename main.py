

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn



# 定义数据转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()  # MNIST的均值和标准差
])

# 加载本地MNIST数据
train_dataset = datasets.MNIST(
    root='./data',  # 数据存储目录
    train=True,              # 训练集
    download=True,  # 自动下载并处理数据集
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',  # 数据存储目录
    train=False,             # 测试集
    download=True,  # 自动下载并处理数据集
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f'训练集样本数: {len(train_dataset)}')
print(f'测试集样本数: {len(test_dataset)}')

class NetWork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1=nn.Linear(784,256)
        self.layer2=nn.Linear(256,10)

    def forward(self,x):
        x=x.view(-1,784)
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.layer2(x)
        return x
    

model=NetWork()
optimizer=torch.optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()

if __name__=="__main__":
    for i in range(10):
        for batch_idx,(data,label) in enumerate(train_loader):
            output=model(data)#神经网络向前传播
            loss=criterion(output,label)#计算output与label之间的损失loss
            loss.backward()#反向传播计算梯度
            optimizer.step()#更新模型参数
            optimizer.zero_grad()#梯度清零

            if batch_idx %100 ==0:
                print(f'Epoch:{i},Batch:{batch_idx},Loss:{loss.item()}')

    torch.save(model.state_dict(),"model.pth") # 保存模型