from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
from RNNModel import Numeric

T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
# d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

features = x[:T-2 ]  # Last example is for Y
labels   = x[1:T-1]

batch_size = 16
n_train = 600
n_train -= n_train%batch_size
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

def get_net_gru(num_hiddens=256):
    # input_size := "feature dimensions"
    rnn_layer = nn.RNN(input_size=1, hidden_size=num_hiddens)
    net = Numeric(rnn_layer, output_size=1)
    return net

net = get_net_gru(256)
# device = d2l.try_gpu()
device='cpu'
net.train(net, train_iter, lr=1, num_epochs=10, device=device)

num_preds=64
preds = net.predict(features[:n_train], num_preds=num_preds, device='cpu')

domain = n_train + num_preds
d2l.plot(
    [time[:domain], time[n_train:domain]],
    [x[:domain]   , preds.detach().numpy()],
    legend=['orig-seq', 'predictions'], xlim=[0, domain],
    figsize=(6,3)
) 

plt.show()
