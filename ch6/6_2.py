from d2l import torch as d2l
import torch
from torch import nn


def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

################################################################################
# Ex. kernel edge detection
X = torch.ones((6, 8))
X[:, 2:6] = 0
# print(X)

# kernel K
K = torch.tensor([[1.0, -1.0]])

Y = corr2d(X, K)
# print(Y)

################################################################################
# Now let's learn that kernel instead

# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = .003

for i in range(400):
    y_hat = conv2d(X)
    l = torch.sqrt(torch.sum((Y - y_hat)**2))
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr*conv2d.weight.grad
    if i%5 == 0:
        print(f'epoch {i}, loss: {l.sum()}')

# [ins] In [65]: print(conv2d.weight.data)
# tensor([[[[ 0.9953, -1.0047]]]])
# we correctly learn the edge-detection kernel

################################################################################
""" 6.2.8. Exercises

1. Construct an image X with diagonal edges.
    1. What happens if you apply the kernel K in this section to it?
    2. What happens if you transpose X?
    3. What happens if you transpose K?

2. When you try to automatically find the gradient for the Conv2D class we created, what kind of error message do you
see?
    ans:I didn't see any in torch.

3. How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel
tensors?

4. Design some kernels manually.
    1. What is the form of a kernel for the second derivative?
    2. What is the kernel for an integral?
    3. What is the minimum size of a kernel to obtain a derivative of degree d?
"""

# 1. Construct an image X with diagonal edges.
from torch import diag, ones

# X = diag(ones(4), -2) + diag(ones(4), 2) + diag(ones(6))
X = diag(ones(6))
"""
tensor([[1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]])
"""


# 1.1. What happens if you apply the kernel K in this section to it?
corr2d(X, K)
"""
tensor([[ 1.,  0.,  0.,  0.,  0.],
        [-1.,  1.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0., -1.,  1.],
        [ 0.,  0.,  0.,  0., -1.]])
"""

# 1.2. What happens if you transpose X?
"""
Nothing:
tensor([[ 1.,  0.,  0.,  0.,  0.],
        [-1.,  1.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0., -1.,  1.],
        [ 0.,  0.,  0.,  0., -1.]])
"""

# 1.3. What happens if you transpose K?
corr2d(X, K.T)
"""
Transpose the output matrix:
tensor([[ 1., -1.,  0.,  0.,  0.,  0.],
        [ 0.,  1., -1.,  0.,  0.,  0.],
        [ 0.,  0.,  1., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  1., -1.,  0.],
        [ 0.,  0.,  0.,  0.,  1., -1.]])
"""

# 3. How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel
# tensors?
# Switch to numpy b/c can't find torch.tile
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = np.array([[1, -1]])

corr2d(X, K)
"""
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
"""

X*torch.tensor(np.tile(K, 4))
""" Close but no cigar:
array([[ 1., -1.,  0., -0.,  0., -0.,  1., -1.],
       [ 1., -1.,  0., -0.,  0., -0.,  1., -1.],
       [ 1., -1.,  0., -0.,  0., -0.,  1., -1.],
       [ 1., -1.,  0., -0.,  0., -0.,  1., -1.],
       [ 1., -1.,  0., -0.,  0., -0.,  1., -1.],
       [ 1., -1.,  0., -0.,  0., -0.,  1., -1.]])
"""

K = diag(ones(8)) + diag(-ones(7), -1)
# gives us the 8x8, but we need an 8x7 to match corr2d -> chop off last column
K = K[:,:-1]
"""Edge Detection Kernel
tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.]])
"""

X@K
"""
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
"""

# 4.2. design an integral kernel
X = torch.tensor([float(i) for i in np.arange(16)]).reshape(4,4)
"""
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
"""

K = diag(ones(4)) + diag(ones(3),1) + diag(ones(2), 2) + diag(ones(1), 3)
""" try an upper-diagonal matrix
tensor([[1., 1., 1., 1.],
        [0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.]])
"""

X@K
""" Shore nuff
tensor([[ 0.,  1.,  3.,  6.],
        [ 4.,  9., 15., 22.],
        [ 8., 17., 27., 38.],
        [12., 25., 39., 54.]])
"""
