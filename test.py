import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn

from main import NetWork



# 定义数据转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()  # MNIST的均值和标准差
])


test_dataset = datasets.MNIST(
    root='./data',  # 数据存储目录
    train=False,             # 测试集
    download=False,  # 自动下载并处理数据集
    transform=transform
)

# 创建数据加载器
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
if __name__ == "__main__":
    model=NetWork()

    model.load_state_dict( torch.load('model.pth'))

    right=0
    for i, (data, label) in enumerate(test_dataset):
        output=model(data)
        predict=output.argmax(1).item()#选择概率最大的标签作为预测结果
        if predict==label:
            right+=1
        else:
            print(f'预测错误，预测值:{predict},真实值:{label},图片数据:{data}')

    sample_num=len(test_dataset)
    acc=right*1.0/sample_num
    print(f'测试集样本数:{sample_num},正确数:{right},准确率:{acc}')
