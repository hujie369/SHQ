import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
 
    def __init__(self, input_size, hidden_size=64, output_size=1, layer=2):
        super(RNN,self).__init__()  # (初始化)
        input_size, hidden_size, output_size = input_size, hidden_size, output_size
        self.rnn = nn.LSTM(
            input_size=input_size,  # d
            hidden_size=hidden_size,         # h 隐藏神经元的个数
            num_layers=layer,           # RNN的层数
            batch_first=True,       # N×L×d，以batch_size为第一维度，若是false则是L×N×d
        )
        self.out = nn.Linear(hidden_size, output_size) # 定义一个全连接层
 
    def forward(self, x):  # 定义前向传播
        x = x.to(torch.float32)
        h0 = torch.zeros(1, x.size(0), 64).to(device)  # h_0的格式为 num_layers×N×h
        r_out, _ = self.rnn(x)
        out = self.out(r_out[:, -1, :])  # 将r_out传给全连接层，-1是最后一层的状态，ht
        return out
    

if __name__ == '__main__':
    input = torch.randn(64,28,4)
    model = RNN(4,64,1)
    output = model(input)
    print(output)