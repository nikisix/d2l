# used an rnn in an n-gram style :(
# THIS IS PROBABLY NOT OPTIMAL

from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
from RNNModel import Numeric

#@tab mxnet, pytorch
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

#@tab mxnet, pytorch
tau = 30
features = d2l.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = d2l.reshape(x[tau:], (-1, 1))


batch_size = 16
n_train = 600
n_train -= n_train%batch_size
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net_mlp(tau):
    net = nn.Sequential(nn.Linear(tau, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

def get_net_gru(τ=1):
    num_hiddens = 256
    rnn_layer = nn.RNN(τ, num_hiddens)
    net = Numeric(rnn_layer, 1)
    # We use a tensor to initialize the hidden state, whose shape is (number of hidden layers, batch size, number of hidden
    # units).
    # initial_state = torch.zeros((1, batch_size, num_hiddens))
    return net


# Square loss
loss = nn.MSELoss()

def train_mlp(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net_mlp(tau)
train_mlp(net, train_iter, loss, 5, 0.01)

# net = get_net_gru(τ=4)
# # device = d2l.try_gpu()
# net.train(net, train_iter, num_preds=1, lr=1, num_epochs=10, device='cpu')

max_steps = 64

#@tab mxnet, pytorch
features = d2l.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# # features[:, i] = d2l.reshape(net(features[:, i - tau: i]), -1)
# # for i in range(tau, tau + max_steps): net.predict(features[:, 0:tau], 2,'cpu')
# # features = net.predict(features[:, 0:tau], 64,'cpu')


steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [d2l.numpy(features[:, tau + i - 1]) for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
plt.show()

"""
Variance drops to 0 after 30 or so iterations into the future
    ipdb> torch.var(features, dim=0)
    tensor([4.9716e-01, 4.9711e-01, 4.9725e-01, 4.9694e-01, 1.5809e-04, 2.3734e-04,
            1.5558e-04, 1.1278e-03, 8.4877e-07, 2.5014e-06, 1.6873e-06, 2.6207e-06,
            1.5340e-08, 5.8878e-09, 1.4902e-08, 6.8168e-09, 1.3024e-10, 1.1096e-11,
            8.0228e-11, 2.0331e-11, 6.0880e-13, 1.1036e-13, 3.6038e-13, 6.7023e-14,
            1.8928e-15, 1.1078e-15, 1.5303e-15, 2.4856e-16, 2.8222e-18, 1.7228e-17,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
            0.0000e+00, 0.0000e+00], grad_fn=<VarBackward1>)

    ipdb> torch.mean(features, dim=0)
    tensor([ 0.2221,  0.2223,  0.2220,  0.2224, -0.2093, -0.2061, -0.2153, -0.2328,
            -0.2119, -0.2130, -0.2128, -0.2111, -0.2121, -0.2120, -0.2120, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121, -0.2121,
            -0.2121, -0.2121, -0.2121, -0.2121], grad_fn=<MeanBackward1>)
"""
