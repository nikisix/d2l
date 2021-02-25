from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

# see chapter for from-scratch implementation

dropout1, dropout2 = 0.2, 0.5
# CONCISE IMPLEMENTATION
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

num_epochs, lr, batch_size = 10, 0.5, 256
net.apply(init_weights)
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()

""" Results:
train_loss:  0.3549059523264567
train_acc:  0.87075 """
