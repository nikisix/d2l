from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = np.array([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = np.array([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return np.stack([corr2d_multi_in(X, k) for k in K], 0)


K = np.stack((K, K + 1, K + 2), 0)
print(f'K-shape: {K.shape}')
corr2d_multi_in_out(X, K)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = np.dot(K, X)  # Matrix multiplication in the fully-connected layer
    return Y.reshape((c_o, h, w))

X = np.random.normal(0, 1, (3, 3, 3))
K = np.random.normal(0, 1, (2, 3, 1, 1))


Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(np.abs(Y1 - Y2).sum()) < 1e-6


# Exercise 6
def corr2d_multi_in_out_2x2(X, K):
    c_i, h, w = X.shape
    c_o, c_ii, kh, kw = K.shape
    assert c_ii == c_i, "Kernel channel dimensions don't match input"
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, kh * kw, c_i))
    Y = np.dot(K, X)  # Matrix multiplication in the fully-connected layer
    Y = Y.sum(axis=2) # input channel dimension
    return Y.reshape((c_o, kh, kw))


K = np.random.normal(0, 1, (2, 3, 2, 2))
Y1 = corr2d_multi_in_out_2x2(X, K)
print(Y1.shape)
Y2 = corr2d_multi_in_out(X, K)
print(Y2.shape)

assert float(np.abs(Y1 - Y2).sum()) < 1e-6

"""Exercises
1. N/A. Wording??

2. Assume an input of shape ci×h×w and a convolution kernel of shape co×ci×kh×kw, padding of (ph,pw), and stride of
(sh,sw)
    2.1. What is the computational cost (multiplications and additions) for the forward propagation?
    2.2. What is the memory footprint?
    2.3. What is the memory footprint for the backward computation?
    2.4. What is the computational cost for the backpropagation?
================================================================================
2.1 Computational Cost (multiplications and additions) for the forward propagation?

Recall (from 6.3):
Convolution Memory Footprint
⌊(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw⌋
    h,w = height, width
    n = input
    k = kernel
    p = padding
    s = stride

Introducing multi-channel outputs to this we end up with:
6.4.2: "To get an output with multiple channels, we can create a kernel tensor of shape ci×kh×kw for _every_ output
channel."

Ignoring padding and stride we end up with:
multiplications:   co * ci * ⌊(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw⌋
additions:         co * ci * ⌊(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw⌋


2.2. What is the memory footprint?
    input: ci nh nw
    conv kernel: co ci kh kw
    output: co kh kw

3.1 By what factor does the number of calculations increase if we double the number of input channels ci and the number
of output channels co?  4
3.2 What happens if we double the padding?  4

4. If the height and width of a convolution kernel is kh=kw=1, what is the computational complexity of the forward
propagation?
Same as fully connected layer. nh*nw

5. Are the variables Y1 and Y2 in the last example of this section exactly the same? Why?
np.dot vs np.sum

6. How would you implement convolutions using matrix multiplication when the convolution window is not 1×1?
???
"""
