import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import device
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 1.定义超参数
Epoch = 1        # 训练次数
Batch_Size = 64  # N，设置了每批次装载(读取)的数据图片为64个（自行设置）
Input_Size = 28  # d，图片宽度
Time_Step = 28   # L，图片长度
LR = 0.01        # 学习率
 
# 2.导入数据集
train_sets = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_sets = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
 
# 3.查看导入的数据集类型
class_name = train_sets.classes  # 查看标签 即[0-9]
class_data_shape = train_sets.data.shape  # torch.Size([60000, 28, 28])
class_target_shape = train_sets.targets.shape  # torch.Size([60000])
 
# 4.数据分批（创建数据集的可迭代对象，也就是说一个batch一个batch的读取数据）
# 关于DataLoader详解见RNN.py文件
train_loader = DataLoader(dataset=train_sets,batch_size=Batch_Size,shuffle=True)
test_loader = DataLoader(dataset=test_sets,batch_size=Batch_Size,shuffle=True)
 
# 5.查看分批导入数据的数据特征
print(type(train_loader))  # 了解一下数据特征，图片是28*28个像素
dataiter = iter(train_loader)  # 创建一个可迭代的对象
imgs, labs = next(dataiter)   # imgs是64张图片的数据，labs则是64张图片的标签
imgs.size()  # torch.Size([64, 1, 28, 28])，1是颜色通道
 
# 6.定义函数，显示一批数据
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # 先转换为numpy格式，颜色通道改变
    mean = np.array([0.485, 0.456, 0.406])  # 均值
    std = np.array([0.229, 0.224, 0.225])   # 标准差
    inp = std * inp + mean  # 数据恢复
    inp = np.clip(inp, 0, 1)  # 数据压缩，像素值限制在【0，1】之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
 
# 7.网格显示 make_grid显示
out = torchvision.utils.make_grid(imgs)
# imshow(out)
 
# 8.定义网络模型
class RNN(nn.Module):
 
    def __init__(self):
        super(RNN,self).__init__()  # (初始化)
        input_size, hidden_size, output_size = Input_Size, 64, 10
        self.rnn = nn.LSTM(
            input_size=Input_Size,  # d
            hidden_size=64,         # h 隐藏神经元的个数
            num_layers=1,           # RNN的层数
            batch_first=True,       # N×L×d，以batch_size为第一维度，若是false则是L×N×d
        )
        self.out = nn.Linear(hidden_size, output_size) # 定义一个全连接层
 
    def forward(self, x):  # 定义前向传播
        h0 = torch.zeros(1, x.size(0), 64).to(device)  # h_0的格式为 num_layers×N×h
        r_out, _ = self.rnn(x)
        out = self.out(r_out[:, -1, :])  # 将r_out传给全连接层，-1是最后一层的状态，ht
        return out
 
# 9.使用定义好的RNN
model = RNN()
# 判断是否有GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型和输入数据都需要to device
model = model.to(device)
 
# 10.定义损失函数，以及分类器 分类问题使用交叉信息熵
loss_func = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
# 11.训练前的参数定义
loss_list = []        # 保存loss
accuracy_list = []    # 保存accuracy
iteration_list = []   # 保存循环次数
iter = 0              # 循坏第0次开始，作为计数器
 
# 12.开始训练
for epoch in range(Epoch):
    # 60000/BATCH_SIZE
    for step, data in enumerate(train_loader):  # STEP∈[0,975]
        model.train()  # 声明训练
        inputs, labels = data  # 取出数据及标签
 
        inputs = inputs.view(-1, 28, 28)  # 修改形状，变换为RNN的输入维度，参数-1 指自动调整size，此时inputs的格式变为[64, 28, 28]
        inputs = inputs.to(device)
        labels = labels.to(device)
 
        optimizer.zero_grad()    # 梯度清零
        outputs = model(inputs)  # 前向传播
        loss = loss_func(outputs, labels)  # 计算损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
 
        # 测试，计算准确率
        if step % 100 == 0:  # 每100次进行一次模型的验证
            model.eval()     # 模型声明
            # 计算验证的accuracy
            correct = 0.0    # 正确的
            total = 0.0      # 总数
            for test_x, test_y in test_loader:
                test_x = test_x.view(-1, 28, 28)  # 验证集，修改形状，变换为RNN的输入维度，参数-1 指自动调整size
                test_x = test_x.to(device)
                # 模型验证
                test_outputs = model(test_x)  # 前向传播
                # 获取预测概率最大值的下标
                predict = torch.max(test_outputs.data, 1)[1]
                # 统计测试集的大小
                total += labels.size(0)
                # 统计判断/预测正确的数量
                correct += (predict == test_y).sum()
            # 计算accuracy
            accuracy = correct / total * 100
            # 保存accuracy，loss，iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("loop：{}，LOSS：{}，Accuracy：{}".format(iter, loss.item(), accuracy))
 
        iter += 1  # 计数器自动加1
 
# 13.可视化
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Loss")
plt.title("LSTM")
plt.show()
 
plt.plot(iteration_list, accuracy_list)
plt.xlabel("Number of Iteration")
plt.ylabel("Accuracy")
plt.title("LSTM")
plt.show()