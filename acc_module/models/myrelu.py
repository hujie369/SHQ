import torch
import torch.nn as nn
class SelfDefinedRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return torch.clamp(inp, 0, 6)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        # return grad_output * torch.clamp(inp, 0)
        return grad_output * torch.where((inp < 0.)+(inp > 6), torch.zeros_like(inp),
                                         torch.ones_like(inp))


class Relu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = SelfDefinedRelu.apply(x)
        return out

def test_myrelu():
    # self defined
    torch.manual_seed(0)

    relu = Relu()  # SelfDefinedRelu
    # inp = torch.randn(5, requires_grad=True)
    inp = torch.tensor([1.,2.,3.,4.,5.], requires_grad=True)
    out = relu((inp).pow(3))

    print(f'Out is\n{out}')

    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nFirst call\n{inp.grad}")

    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")

    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")

def test_relu():
    # torch defined
    torch.manual_seed(0)
    # inp = torch.randn(5, requires_grad=True)
    inp = torch.tensor([1.,2.,3.,4.,5.], requires_grad=True)
    out = torch.nn.functional.relu6((inp).pow(3))

    print(f'Out is\n{out}')

    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nFirst call\n{inp.grad}")

    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")

    inp.grad.zero_()
    out.backward(torch.ones_like(inp), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")

if __name__ == '__main__':
    test_myrelu()
    test_relu()
    # a = torch.tensor(7)
    # x = torch.clamp(a, 0)
    # print(x)
    # a = torch.tensor([1,2,3,4,5,6])
    # print((a>0) * (a<2))