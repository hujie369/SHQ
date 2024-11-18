import torch
import torch.nn as nn
class SelfDefinedRelu1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return torch.clamp(inp, 0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        # return grad_output * torch.clamp(inp, 0)
        return grad_output * torch.where((inp < 0.)+(inp > 1), torch.zeros_like(inp),
                                         torch.ones_like(inp))


class Relu1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = SelfDefinedRelu1.apply(x)
        return out
    

class conv1d(torch.nn.Module):
    def __init__(self, in_dim, out, kernel=5, stride=1, padding=1, pool=True):
        super(conv1d, self).__init__()
        self.in_dim = in_dim
        self.out = out
        self.layer = torch.nn.Conv1d(in_dim, out, kernel, stride, padding)
        self.bn = torch.nn.BatchNorm1d(self.out)
        self.max = torch.nn.MaxPool1d(2)
        self.pool = pool

    def forward(self, input_data):
        input_data = input_data.to(torch.float32)
        if self.pool == True:
            x = self.max(self.bn(self.layer(input_data)))
        else:
            x = self.bn(self.layer(input_data))
        # print(x.shape)
        return x
    
class classifier(torch.nn.Module):
    def __init__(self, in_channel, act='sigmoid'):
        super(classifier, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_channel, 10),
            torch.nn.Sigmoid(),
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(10, 1)
        )
        
        # self.layer = torch.nn.Linear(in_channel, 10)
        # if act == 'sigmoid':
        #     self.act = torch.nn.Sigmoid()
        # elif act == 'relu':
        #     self.act = torch.nn.ReLU()
        # elif act == 'relu1':
        #     self.act = Relu1()
        # self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, input_data):
        input_data = input_data.to(torch.float32)
        # x = self.act(self.bn(self.layer1(input_data)))
        x = self.layer(input_data)
        return x



class CNN(torch.nn.Module):
    def __init__(self,in_dim,layers,act='sigmoid'):
        super(CNN, self).__init__()
        self.in_dim=in_dim
        self.act = act
        self.hidden1 = 14
        self.hidden2 = 28
        self.hidden3 = 56
        self.hidden4 = 112
        self.layers = layers

        if layers == 1:
            self.point = 13
            self.layer = nn.Sequential(
                conv1d(self.in_dim, self.hidden1),  # 14*13
            )
            self.linear = classifier(self.hidden1*self.point)

        elif layers == 2:
            self.point = 6
            self.layer = nn.Sequential(
                conv1d(self.in_dim, self.hidden1, kernel=5),  # 14*13
                conv1d(self.hidden1, self.hidden2, kernel=4)  # 28*6
            )
            self.linear = classifier(self.hidden2*self.point)

        elif layers == 3:
            self.point = 2
            self.layer = nn.Sequential(
                conv1d(self.in_dim, self.hidden1, kernel=5),  # 14*13
                conv1d(self.hidden1, self.hidden2, kernel=4),  # 28*6
                conv1d(self.hidden2, self.hidden3, kernel=5)  # 56*2
            )
            self.linear = classifier(self.hidden3*self.point)

        # elif layers == 4:
        #     self.point = 1
        #     self.layer = nn.Sequential(
        #         conv1d(self.in_dim, self.hidden1, kernel=3),  # 14*14
        #         conv1d(self.hidden1, self.hidden2, kernel=3),  # 28*7
        #         conv1d(self.hidden2, self.hidden3, kernel=4),  # 56*3
        #         conv1d(self.hidden3, self.hidden4, kernel=3, padding=0, pool=False)  # 112*1
        #     )
        #     self.linear = classifier(self.hidden4*self.point)

        elif layers == 4:
            self.point = 2
            self.layer = nn.Sequential(
                conv1d(self.in_dim, self.hidden1, kernel=5),  # 14*13
                conv1d(self.hidden1, self.hidden2, kernel=4),  # 28*6
                conv1d(self.hidden2, self.hidden3, kernel=3, pool=False),  # 56*6
                conv1d(self.hidden3, self.hidden3, kernel=5),  # 56*2
            )
            self.linear = classifier(self.hidden3*self.point)
        
        #pass

    def forward(self, x):
        if self.layers == 1:
            x = self.layer(x)
            x = x.view(-1, self.hidden1*self.point)
            x = self.linear(x)
        elif self.layers == 2:
            x = self.layer(x)
            x = x.view(-1, self.hidden2*self.point)
            x = self.linear(x)
        elif self.layers == 3:
            x = self.layer(x)
            x = x.view(-1, self.hidden3*self.point)
            x = self.linear(x)
        elif self.layers == 4:
            x = self.layer(x)
            x = x.view(-1, self.hidden3*self.point)
            x = self.linear(x)
        
        return x

if __name__ == '__main__':
    input = torch.randn(2,1,28)
    model = CNN(1,3)
    output = model(input)
    print(output.shape)
    # print(model)