import numpy as np
import pandas as pd
import torch
from torch import nn
from models.linear import Linear
from models.rnn import RNN
from models.cnn import CNN
from torch import device
from dataset import load_acc, dataloader
import torch.utils.data as data

def bit2code(bit):
    '''对每个量化位宽进行热编码'''
    if bit == 2.:
        return [0.,0.,0.,1.]
    elif bit == 4.:
        return [0.,0.,1.,0.]
    elif bit == 6.:
        return [0.,1.,0.,0.]
    else:
        return [1.,0.,0.,0.]

def train(epoch_num):
    # 循环外可以自行添加必要内容
    mode = True
    model.train(mode=mode)
    #print(data_loader_train)
    total_loss = 0
    cnt = 0
    for index,(stragety,true_labels) in enumerate(data_loader_train):
        #这里用index和 enumerate，为了统计训练到了第几个样本，中间可以返回过程值
        stragety = stragety.to(device)
        true_labels = true_labels.to(device)
        optimizer.zero_grad()
        output = model(stragety)
        
        true_labels = true_labels.squeeze()
        output = output.squeeze()
        loss = loss_function(output.float(), true_labels)#得到损失函数
        total_loss = total_loss + loss
        cnt = cnt + 1
        loss.backward() # 反向传播训练参数
        optimizer.step()
        # 必要的时候可以添加损失函数值的信息，即训练到现在的平均损失或最后一次的损失，下面两行不必保留
        
    print(f'epoch:{epoch_num}, average_{args.loss}loss:{total_loss/cnt}')  # 获取损失
    return total_loss/cnt


def validation(verbose=False):
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    total_loss = np.array([])
    accuracy = 0
    val_loss_function = torch.nn.L1Loss(reduction='none')
    # reg to compute L2
    mean = 0
    std = 0
    with torch.no_grad():  
        for data in data_loader_val:
            stragety, true_labels = data
            stragety = stragety.to(device)
            true_labels = true_labels.to(device)
            output = model(stragety)
            output = output.squeeze()
            true_labels = true_labels.squeeze()
            
            loss = val_loss_function(output.float(), true_labels)
            # reg = reg.append(loss)
            
            total_loss = np.append(total_loss, loss.cpu().numpy())
        accuracy = np.mean(total_loss)    
        if verbose == True:
            print('output-true_labels:',output-true_labels)
            
            
            std = np.std(total_loss)
            print('val_mean:',accuracy)
            print('val_std',std)
            #print(images, true_labels)
            #pass
    # accuracy = total_loss 
    # print("     ", "预测MSE_loss:", accuracy)
    print("     准确度平均预测误差:{:.4f}".format(accuracy))
    return accuracy
    

# def alltest(result_path):
#     # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
#     loss_list = []
#     acc_list = []
#     with torch.no_grad():
#         #f = open('E:\桌面\dml\深度学习课程-实验1\\result.txt','w') #写对应文件,这里用'w'方法，覆盖原文件
#         f = open(result_path,'w')
#         for data in data_loader_test:
#             images,label= data
#             output = model(images.reshape(-1,784)) 
#             pred = output.data.max(dim=-1)[-1]  # max(dim=-1)：行角度找最大值，dim=0：列最大值，返回下标
#             f.write(str(pred.item())+'\n')#将结果写入文档
#         f.close()
#     # 将结果按顺序写入txt文件中，下面一行不必保留
#     pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='choose FC or CNN or RNN or RL', type=str, default='CNN')
    parser.add_argument('-lay', help='the number of layers', type=int, default=3)
    parser.add_argument('-batchsize', type=int, default=128)
    parser.add_argument('-numworker', type=int, default=8)
    parser.add_argument('-verbose', help='print some result or not', type=bool, default=True)
    parser.add_argument('-epoch', type=int, default=60)
    parser.add_argument('-loss', help='which loss function to use', type=str, default='L1')
    parser.add_argument('-train2imagenet', help='train for imagenet or not', type=bool, default=True)
    args = parser.parse_args()

    print(f'using {args.model}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 读取acc_all数据
    # x 为 stragety , y 为 acc
    if args.train2imagenet == False:
        x, y = load_acc.openreadtxt('./dataset/acc_all')
    else:
        x, y = load_acc.openreadtxt('./dataset/imagenet/imagenet3k')

    in_features = x.shape[1]  # 28
    # print(in_features)

    # 如果是RNN对x进行编码
    if args.model == 'RNN':
        rnn_x =[]
        for stragety in x:
            rnn_stragety =[]
            for i in range(len(stragety)):
                rnn_stragety.append(bit2code(stragety[i]))
            # print(rnn_stragety)
            rnn_x.append(rnn_stragety)
        rnn_x = np.array(rnn_x,dtype=float)
        x = rnn_x
    elif args.model == 'CNN':
        x = x.reshape(-1,1,28)
        

    # 构建数据集
    dataset_train = dataloader.Mydataset(x, y, 'train')
    dataset_val = dataloader.Mydataset(x, y, 'val')
    
    # 构建数据加载器
    data_loader_train = data.DataLoader(dataset_train, batch_size=args.batchsize, num_workers=args.numworker)
    data_loader_val = data.DataLoader(dataset_val, batch_size=args.batchsize, num_workers=args.numworker)
    
    #print(data_loader_train)
    # 初始化模型对象，可以对其传入相关参数
    if args.model == 'FC':
        model = Linear(in_features, 1, args.lay -1, 'sigmoid')
    elif args.model == 'RNN':
        model = RNN(4,64,1,args.lay)
    elif args.model == 'CNN':
        model = CNN(1,args.lay)
    
    state_dict = torch.load(f'./output/{args.model}.pth')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)

    # 损失函数设置
    if args.loss == 'MSE':
        loss_function = torch.nn.MSELoss()  # torch.nn中的损失函数进行挑选，并进行参数设置
    elif args.loss == 'L1':
        loss_function = torch.nn.L1Loss()
    # if torch.cuda.is_available():	
    #     loss_function = loss_function.cuda()
    # 优化器设置
    # relu
    optimizer =torch.optim.Adam(model.parameters(), lr = 0.001) # torch.optim中的优化器进行挑选，并进行参数设置
    # relu1
    # optimizer =torch.optim.Adam(model.parameters(), lr = 0.00005)
    max_epoch = args.epoch  # 自行设置训练轮数
    num_val = 1  # 经过多少轮进行验证

    trainlossdata = []
    vallossdata = []
    # 然后开始进行训练
    best = 1.
    for epoch in range(max_epoch):
        trainlossdata.append(float(train(epoch)))
        # trainlossdata.append(float(train(epoch))**0.5)
        # 在训练数轮之后开始进行验证评估
        if epoch % num_val == 0:
            val = float(validation(args.verbose))
            vallossdata.append(float(val))
            if val < best:
                best = val
                torch.save(model, f'./output/{args.model}_train2i.pth')
            

    # 绘图
    # from tools.pic import pic_loss
    # pic_loss(trainlossdata, vallossdata, max_epoch)


    # val
    # val = float(validation(args.verbose))
    # print('val:',val)
    # print(model)

