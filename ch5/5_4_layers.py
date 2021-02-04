import torch
from torch import nn
from torch.nn import functional as F

###############################################################################
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
Y = net(torch.rand(4, 8))
print(Y.mean())

################################################################################
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)
print(dense.weight)
print(dense(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))

"""
Exercises
1. Design a layer that takes an input and computes a tensor reduction, i.e.,
  it returns y_k = ∑_i,j(W_ijk x_i x_j)
2. Design a layer that returns the leading half of the Fourier coefficients of the data.
"""

# 1. Design a layer that takes an input and computes a tensor reduction, i.e.,
#   it returns y_k = ∑_i,j(W_ijk x_i x_j)

class ReductionLayer(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
    def forward(self, X):
        return torch.sum(X*self.weight)

red = ReductionLayer(3,3)
red(torch.rand(3,3))


class FFT_Layer(nn.Module):
    """5_4_2. Design a layer that returns the leading half of the Fourier coefficients of the data.
    Parameterless"""
    def __init__(self, in_units, units):
        super().__init__()
    def forward(self, X):
        half = int(X.shape[1]/2)
        return torch.fft.fft(X)[:, :half]

fft_net = FFT_Layer(3,3)
fft_net(torch.rand(3,3))
