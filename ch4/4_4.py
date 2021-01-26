from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt

max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
# for i in range(max_degree):
    # poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!

for i in range(max_degree):
    poly_features[:, i] = math.log(poly_features[:, i], i)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [torch.tensor(x, dtype=\
    torch.float32) for x in [true_w, features, poly_features, labels]]

# features[:2], poly_features[:2, :], labels[:2]


def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    final_training_loss = evaluate_loss(net, train_iter, loss)
    print(f'final training loss: {final_training_loss}')
    print('weight:', net[0].weight.data.numpy())
    return final_training_loss


# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
training_loss = {}
train(poly_features[:n_train, :6], poly_features[n_train:, :6],
          labels[:n_train], labels[n_train:])

plt.show()

"""
4.4.6. Exercises

    1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.

    2. Consider model selection for polynomials:

        1. Plot the training loss vs. model complexity (degree of the polynomial).
        2. What do you observe?
        3. What degree of polynomial do you need to reduce the training loss to 0?
        4. Plot the test loss in this case.
        5. Generate the same plot as a function of the amount of data.

    3.1 What happens if you drop the normalization (1/i!) of the polynomial features x_i

? 3.2 Can you fix this in some other way?

4. Can you ever expect to see zero generalization error?
-----------------------------------------------------------------------------
1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.
    from text:
    𝐰∗ = (𝐗⊤𝐗)−1𝐗⊤𝐲.
"""
"""
X = poly_features[:, :4]

torch.matmul(
    torch.inverse(torch.matmul(X.t(), X)),
    torch.matmul(X.t(), labels)
)
"""

# Out[73]: tensor([ 4.9926,  1.1961, -3.3868,  5.5776])

# 2.1. Plot the training loss vs. model complexity (degree of the polynomial).
"""
for i in range(1, 5):
    training_loss[i] = train(poly_features[:n_train, :i], poly_features[n_train:, :i],
          labels[:n_train], labels[n_train:])

plt.plot(training_loss.keys(), training_loss.values())
"""

# 2.2 What do you observe?
# ans: training loss goes down as model complexity increases

# 2.3. What degree of polynomial do you need to reduce the training loss to 0?
# ans: 3rd degree

# 3.1 What happens if you drop the normalization (1/i!) of the polynomial features x_i
# OVERFIT: test error never quite drops to the level of training error.
