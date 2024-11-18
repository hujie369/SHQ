import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import copy
import torchvision
# from pytorch_ssim import ssim

from energy import memMacs_linear, memMacs_conv, network_energy


def gradscale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y

def roundpass(x):
    yout = torch.round(x)
    ygrad = x
    y = (yout - ygrad).detach() + ygrad
    return y

def myquantize(x, s, z, qmin, qmax):
    # weight'shape: (outc, inc ,ks1, ks2)
    # act's shape: (bs, c, h, w)
    # s and z: (1)
    N = torch.tensor(x.numel(), dtype=torch.float, device=x.device)

    gradScaleFactor = torch.rsqrt((qmax - qmin) * N)
    s = gradscale(s, gradScaleFactor)
    x = x / s + z
    x = torch.clamp(x, qmin, qmax)
    x = roundpass(x) - z
    x = x * s
    return x


# uniform-quantizer
class QModule(nn.Module):
    def __init__(self, w_bit=8, a_bit=8):
        super(QModule, self).__init__()

        self.w_bit = w_bit
        self.w_min = -2**(w_bit - 1)
        self.w_max = 2**(w_bit - 1) - 1
        self.w_s = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float), requires_grad=True)
        self.w_z = torch.nn.Parameter(torch.tensor([0], dtype=torch.int), requires_grad=False)
        
        self.a_bit = a_bit
        self.a_min = 0
        self.a_max = 2**a_bit - 1
        self.a_s = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float), requires_grad=True)
        self.a_z = torch.nn.Parameter(torch.tensor([0], dtype=torch.int), requires_grad=False)
        
        self.b_bit = w_bit + a_bit
        self.eeval = False

    def _quantize(self, inputs, weight, bias):
        inputs = myquantize(inputs, self.a_s, self.a_z, self.a_min, self.a_max)
        weight = myquantize(weight, self.w_s, self.w_z, self.w_min, self.w_max)
        # bias = self._quantize_bias(bias=bias)
        return inputs, weight, bias


class QConv2d(QModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 w_bit=8, a_bit=8):
        super(QConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        out =  F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        # 能耗评估
        if self.eeval:
            self.params, self.MACs = memMacs_conv(inputs.shape, weight.shape, out.shape, self.w_bit, self.a_bit)

        # 提取中间结果
        if hasattr(self, 'split') and self.split == True:
            self.tmp = out
        return out


class QLinear(QModule):
    def __init__(self, in_features, out_features, bias=True, w_bit=8, a_bit=8):
        super(QLinear, self).__init__(w_bit=w_bit, a_bit=a_bit)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        out =  F.linear(inputs, weight=weight, bias=bias)
        # 能耗评估
        if self.eeval:
            self.params, self.MACs = memMacs_linear(inputs.shape, out.shape, self.w_bit, self.a_bit)
        # 提取中间结果
        if hasattr(self, 'split') and self.split == True:
            self.tmp = out
        return out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


def set_fix_weight(model, fix_weight=True):
    if fix_weight:
        print('==> set weight fixed')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_fix_weight(fix_weight=fix_weight)


def build_index(qmodel, quantizable_type=[QConv2d, QLinear]):
    quantizable_idx = []
    for i, m in enumerate(qmodel.modules()):
        if type(m) in quantizable_type:
            quantizable_idx.append(i)
            # print(f'{i}:{m}')
    # print('#####################')
    # print(quantizable_idx)
    # print(quantizable_type)
    return quantizable_idx


def uniform_quantize(qnet, w_bit=8, a_bit=8, skip_first=True):
    clear_qnet(qnet)
    idx = build_index(qnet)
    if skip_first:
        idx = idx[1: ]
    for i, layer in enumerate(qnet.modules()):
        if i not in idx:
            continue
        else:
            layer.w_bit = w_bit
            layer.a_bit = a_bit


def mixed_quantize_with_partition(qmodel, strategy, split, q_idx=None, a_bit=8):
    '''
    重置所有策略
    model.split = split
    首层不量化，activation统一8位(除了第一层)，分隔层被split=True
    split_idx in [-1, 0, 1, 2, ..., len(model)-1],  -1表示原始输入传云端，模型不量化，没有层被标记
    split = 0，由于跳过首层，模型也不量化，但第一层会被split标记
    '''
    clear_qnet(qmodel)
    qmodel.split = split
    q_idx = q_idx if q_idx is not None else build_index(qmodel)
    if split < 0:
        return
    assert len(q_idx[1: ]) == len(strategy), \
         'You should provide the same number of bit setting as layer list for weight quantization!'
    q_dict = {n: b for n, b in zip(q_idx[1: split+1], strategy)}

    for i, layer in enumerate(qmodel.modules()):
        if i == q_idx[split]:
            layer.split = True
        if i not in q_dict.keys():
            continue
        else:
            layer.w_bit = q_dict[i]
            layer.a_bit = a_bit


def mixed_quant_with_partition_bw(qmodel, strategy, split, block_type, a_bit=8, acc_module=False):
    '''
    重置所有策略
    split in [0, lenQModel(qmodel, block_type)]
    '''
    wei2act = {8:8, 6:6, 4:6, 2:4, 32:32}
    clear_qnet(qmodel)
    q_idx = build_index(qmodel)
    if split < 0:
        return
    assert len(q_idx[: ]) == len(strategy), \
         'You should provide the same number of bit setting as layer list for weight quantization!'
    q_dict = {n: b for n, b in zip(q_idx[:], strategy)}
    next_block_idx = bw_split(qmodel, block_type, split)
    next_block_idx = next_block_idx if next_block_idx != -1 else 100000000000000
    qmodel.next_block_idx = next_block_idx

    count = 0
    for i, layer in enumerate(qmodel.modules()):
        if i in q_dict.keys() and i < next_block_idx:
            count += 1
            layer.w_bit = q_dict[i]
            layer.a_bit = wei2act[q_dict[i]]

        # split修改
        if i in q_dict.keys() and i >= next_block_idx:
            layer.w_bit = 8  # 默认8bits
            layer.a_bit = a_bit


def bw_split(qmodel, block_type, split):
    '''
    从第split个block分开，返回下一block的idx，如果没有则返回-1
    split in [0, ...]
    '''
    block_idx = build_index(qmodel, block_type)
    for i, m in enumerate(qmodel.modules()):
        if i == block_idx[split]:
            m.split = True
    return block_idx[split+1] if split + 1 < len(block_idx) else -1


def load_qnet(qmodel, path):
    ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
    qmodel.load_state_dict(ch, strict=False)


def set_eeval(qnet, eeval, types=[QConv2d, QLinear]):
    for module in qnet.modules():
        if type(module) in types:
            module.eeval = eeval


def energy_eval(qnet, input_shape, device='cpu'):
    '''
    qnet should be mixedquantized first 
    input_shape (3, 32, 32) like 3-d 
    will set_eeval(False) automatically 
    split_idx in [-1, 0, 1, 2, ..., len(model)-1],  -1表示原始输入传云端，本地无能耗
    '''
    with torch.no_grad():
        split = qnet.split
        set_eeval(qnet, True)
        qnet.to(device)(torch.randn(1, *input_shape).to(device))
        e  = extract(qnet)
        set_eeval(qnet, False)
        e_sum = 0.0
        edge_idx = build_index(qnet)[0: split+1]
        for k in e.keys():
            if k in edge_idx:
                e_sum += e[k]
        e_sum = e_sum if e_sum != 0 else network_energy(torch.randn(1, *input_shape).to(device), num_bit=32)
    return e_sum


def energy_eval_bw(qnet, input_shape, device='cpu'):
    with torch.no_grad():
        # next_block_idx = qnet.next_block_idx
        set_eeval(qnet, True)
        qnet.to(device)(torch.randn(1, *input_shape).to(device))
        total_params, total_MACs = extract(qnet)
        set_eeval(qnet, False)
        params_sum = 0.0
        MACs_sum = 0.0
        count = 0
        for k in total_params.keys():
            # if k < next_block_idx:
            #     count += 1
            #     e_sum += e[k]
            params_sum += total_params[k]
            MACs_sum += total_MACs[k]
        # e_sum = e_sum if e_sum != 0 else network_energy(torch.randn(1, *input_shape).to(device), num_bit=32)
        # print(f'{count} layers')
    return params_sum, MACs_sum



def extract(qnet, types=[QConv2d, QLinear]):
    total_params = {}
    total_MACs = {}
    for i, module in enumerate(qnet.modules()):
        if type(module) in types and hasattr(module, 'params'):
            total_params.update({i: module.params})
        if type(module) in types and hasattr(module, 'MACs'):
            total_MACs.update({i: module.MACs})
    return total_params, total_MACs


def load_qnet(qnet, path):
    ch = {n.replace('module.', ''): v for n, v in torch.load(path).items()}
    # ch = {n.replace('module.', ''): v for n, v in torch.load(path).stem()}
    qnet.load_state_dict(ch, strict=False)
    # qnet = torch.load(path)
    return qnet


def extract_IR(qnet):
    '''
    提取IR，如果qnet.split_idx = -1即不分隔则返回None
    '''
    res = None
    for module in qnet.modules():
        if hasattr(module, 'tmp'):
            # print(module)
            res = module.tmp
    return res


def clear_qnet(qnet):
    '''
    重置量化策略 + 删除tmp
    '''
    if hasattr(qnet, 'split'):
        del qnet.split
    for module in qnet.modules():
        if hasattr(module, 'tmp'):
            del module.tmp
        if hasattr(module, 'split'):
            del module.split
        if type(module) in  [QConv2d, QLinear]:
            module._a_bit = -1
            module.w_bit = -1


def privacy_eval(qnet, sample, label, device='cpu', test_sample=False):
    with torch.no_grad():    
        # sample, label = iter(dataLoader).next()
        shape = sample.shape[-2: ]
        qnet, sample, label = qnet.to(device), sample.to(device), label.to(device)

        qnet.eval()
        output = qnet(sample)
        if test_sample:
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            print(f'top1: {correct[:, :1].sum() / sample.shape[0]}')

        ir = extract_IR(qnet)
        ir_images = [sample]
        if ir is not None:
            # print(ir.shape, end=' ')
            ir_images = ir2images(ir, shape=shape, device=device)
        return minMSE(ir_images, sample)
        # return minKL(ir_images, sample, qnet)


# def maxSSIM(ir_images, input):
#     ssims = []
#     for ir_image in ir_images:
#         ssims.append(ssim(ir_image, input))
#     return max(ssims)

def minKL(ir_images, input, model):
    kls = []
    output = F.softmax(model(input))
    for ir_image in ir_images:
        output1 = F.softmax(model(ir_image))
        kls.append(kl_divergence(output, output1) / output.shape[0])
    return min(kls)


def kl_divergence(p, q):
    '''
    p, q首先需要进行softmax处理
    '''
    return (p * (torch.log(p) - torch.log(q))).sum()


def ir2images(ir, shape, device='cpu', normalized=False, mean=None, std=None):
    '''
    将IR映射为图片张量
    如：(64, 64, 14, 14)的IR 将被映射为64个 (64, 3, 32, 32)张量
    '''
    ir_images = []
    if ir.shape[1] == 3:
        return [ir]
    for i in range(ir.shape[1]):
        x = ir[:, i]
        x = torchvision.transforms.functional.resize(torch.stack([x, x, x], dim=1), shape)
        if normalized:
            x = torchvision.transforms.functional.normalize(x, mean=mean, std=std)
        ir_images.append(x.to(device))
    return ir_images


def lenQmodel(model, layer_type=[QConv2d, QLinear]):
    return len(build_index(model, layer_type))


def mse(img1, img2):
    return torch.mean(torch.square(img1 - img2))


def minMSE(ir_images, input, device='cuda'):
    mses = []
    for ir in ir_images:
        mses.append(mse(ir.to(device), input.to(device)))
    return min(mses)