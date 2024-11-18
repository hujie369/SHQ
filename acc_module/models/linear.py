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
    

class linear(torch.nn.Module):
    def __init__(self, in_dim, out, act):
        super(linear, self).__init__()
        self.in_dim = in_dim
        self.out = out
        self.layer = torch.nn.Linear(self.in_dim, self.out)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'relu1':
            self.act = Relu1()
        self.bn = torch.nn.BatchNorm1d(self.out)

    def forward(self, input_data):
        input_data = input_data.to(torch.float32)
        x = self.act(self.bn(self.layer(input_data)))
        return x



class Linear(torch.nn.Module):
    def __init__(self,in_dim,out,layers,act):
        super(Linear, self).__init__()
        self.in_dim=in_dim
        self.act = act
        
        if layers == 1:
            self.hidden1 = 14

            self.layer = nn.Sequential(
                linear(self.in_dim, self.hidden1, act=self.act),
                linear(self.hidden1, out, act=self.act)
            )

        if layers == 2:
            self.hidden1 = 14
            self.hidden2 = 7

            self.layer = nn.Sequential(
                linear(self.in_dim, self.hidden1, act=self.act),
                linear(self.hidden1, self.hidden2, act=self.act),
                linear(self.hidden2, out, act=self.act)
            )
        elif layers == 3:
            self.hidden1 = 20
            self.hidden2 = 14
            self.hidden3 = 6

            self.layer = nn.Sequential(
                linear(self.in_dim, self.hidden1, act=self.act),
                linear(self.hidden1, self.hidden2, act=self.act),
                linear(self.hidden2, self.hidden3, act=self.act),
                linear(self.hidden3, out, act=self.act)
            )

            
        

        #pass

    def forward(self, input_data):
        # 此处添加模型前馈函数的内容，return函数需自行修改
        x = self.layer(input_data)
        return x

if __name__ == '__main__':
    input = torch.randn(64,28)
    model = Linear(28,1,3,'sigmoid')
    output = model(input)
    print(output.shape)