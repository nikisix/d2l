from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=.1)

net.apply(init_weights);


batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
# plt.savefig("mygraph.png")


"""
4.3.3. Exercises

    1. Try adding different numbers of hidden layers (you may also modify the learning rate). What setting works best?

    2. Try out different activation functions. Which one works best?

    3. Try different schemes for initializing the weights. What method works best?
    ---------------------------------------------------------------------------------

    Baseline Accuracy:
        train & test ~= 85%
        train loss ~= 38%

    2. Sigmoid => train_loss ~= 50%
    3. Softmax at the end -> trainloss ~= 1.6825744838078818
    4. Tanh activation function slightly worse than ReLU:
        83% training error
        40% training loss

    5. Initiliazing the weights with mean=1 leads to a high training loss: AssertionError: 2.8205257770538332
    6. mean=0, std=.1 yeilds better results, training loss ~= 35%
    7. std=1 -> high training loss: AssertionError: 0.7331619616508483
    8. adding another hidden layer (256, 256) leads to dramatically improved performance

"""
