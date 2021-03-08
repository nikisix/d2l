from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
from RNNModel import Numeric

T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
ax = plt.axes()
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3), axes=ax)

batch_size = 16
train_seq_len, pred_seq_len = 360, 360
n_train = 500
n_train -= n_train%batch_size

τ = train_seq_len
features = d2l.zeros((T - τ, τ))
labels   = d2l.zeros((T - τ, τ))
for i in range(τ):
    features[:, i] = x[i    : T - τ + i]
    labels[:, i]   = x[i + 1: T - τ + i + 1]

# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

def get_net_gru(hidden_size, input_size, output_size):
    # input_size := "feature dimensions"
    rnn_layer = nn.RNN(input_size, hidden_size)
    net = Numeric(rnn_layer, output_size)
    return net

net = get_net_gru(256, train_seq_len, pred_seq_len)
# device = d2l.try_gpu()
device='cpu'
net.train(net, train_iter, lr=1, num_epochs=10, device=device)

preds = net.predict(
                prefix=features[n_train-5: n_train],
                prediction_seq=features[n_train+1],
                device='cpu')

domain = n_train + pred_seq_len
d2l.plot(
    [time[:domain], time[n_train:domain]],
    [x[:domain]   , preds.reshape(pred_seq_len).detach().numpy()],
    legend=['orig-seq', 'predictions'], xlim=[0, domain],
    figsize=(6,3), axes=ax
) 

plt.show()
