
"""NOTES

L1 VS L2 REGULARIZATION
Moreover, you might ask why we work with the L2 norm in the first place and not, say, the L1 norm. In fact, other
choices are valid and popular throughout statistics. While L2-regularized linear models constitute the classic ridge
regression algorithm, L1-regularized linear regression is a similarly fundamental model in statistics, which is
popularly known as lasso regression.

One reason to work with the L2 norm is that it places an outsize penalty on large components of the weight vector. This
biases our learning algorithm towards models that distribute weight evenly across a larger number of features. In
practice, this might make them more robust to measurement error in a single variable. By contrast, L1 penalties lead to
models that concentrate weights on a small set of features by clearing the other weights to zero. This is called feature
selection, which may be desirable for other reasons.

TLDR: L2 more evenly distributes weights. L1 concentrates weight on a small number of features.

"""

from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# λ
def train(λ):
    w, b = init_params()
    # net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    net = lambda X: torch.matmul(X,w) + b
    loss = lambda y_hat, y: (y_hat - y)**2

    num_epochs, lr = 20, .01
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with(torch.enable_grad()):
                l = (y - (torch.matmul(X,w) + b))**2 + λ*l2_penalty(w)
                l.sum().backward()
                sgd([w, b], lr, batch_size)
        # print(l.sum()/batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())

train(2)
train(4)
train(10)
train(20)
plt.show()
