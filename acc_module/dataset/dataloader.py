import torch.utils.data as data
import numpy as np



class Mydataset(data.Dataset):

    def __init__(self, x, y, mode):
        self.x = x
        self.y = y
        
        
        # 对数据集进行划分
        if mode == "train": # 80%
            self.x = self.x[:int(0.8*len(self.x))]
            self.y = self.y[:int(0.8*len(self.y))]
        elif mode == "val": # 80~100%
            self.x = self.x[int(0.8*len(self.x)):]
            self.y = self.y[int(0.8*len(self.y)):]
        else: 
            print('mode is false')
        
        self.idx = list()
        for item in self.x:
            self.idx.append(item)

        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


if __name__ ==('__main__'):
    x = np.array(range(100)).reshape(10, 10) # 模拟输入， 10个样本，每个样本长度为10
    y = np.array(range(10))  # 模拟对应样本的标签， 10个标签
    datasets = Mydataset(x, y, 'val')  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=64, num_workers=8) 

    for i, (input_data, target) in enumerate(dataloader):
        print('input_data%d' % i, input_data)
        print('target%d' % i, target)


