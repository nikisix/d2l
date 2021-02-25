from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

""" Batch Normalization """

""" APPLYING BATCH NORM IN LeNet """

""" CONCISE IMPLEMENTATION """
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 10.0, 5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
plt.show()

""" 7.5.8. Exercises
1  Can we remove the bias parameter from the fully-connected layer or the convolutional layer before
the batch normalization? Why?

2  Compare the learning rates for LeNet with and without batch normalization.
    Plot the increase in training and test accuracy.
    How large can you make the learning rate?

3  Do we need batch normalization in every layer? Experiment with it?

4  Can you replace dropout by batch normalization? How does the behavior change?

5  Fix the parameters beta and gamma, and observe and analyze the results.

6  Review the online documentation for BatchNorm from the high-level APIs to see the other
    applications for batch normalization.

7  Research ideas: think of other normalization transforms that you can apply?
    Can you apply the probability integral transform?
    How about a full rank covariance estimate?
"""

""" 1  Can we remove the bias parameter from the fully-connected layer or the convolutional
    layer before the batch normalization? Why?
Yes to both because the batch normalization has its own bias parameter. """

""" 2  Compare the learning rates for LeNet with and without batch normalization.
    Plot the increase in training and test accuracy.
    How large can you make the learning rate?

LeNet without batch normalization:
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

At learning_rate = 1
    WITH
        loss 0.256, train acc 0.905, test acc 0.815
        51744.3 examples/sec on cuda:0

    WITHOUT
        loss 0.451, train acc 0.829, test acc 0.823
        64655.0 examples/sec on cuda:0

At learning_rate = 10
    WITH (and get's there FAST)
        loss 0.279, train acc 0.896, test acc 0.851
        51624.0 examples/sec on cuda:0

    WITHOUT
        loss 2.320, train acc 0.100, test acc 0.100
        64170.5 examples/sec on cuda:0

At learning_rate = 100
    WITH - blows up even with batch normalization
"""

""" 3  Do we need batch normalization in every layer? Experiment with it?
Removed all but the FIRST BatchNorm and learning rate == 10 blew up.
Removed all but the LAST BatchNorm and learning rate == 10 blew up.
Removed all but the FIRST AND LAST BatchNorm and learning rate == 10 blew up.
=>
Yes, we need Batch Normalization at every level.
"""

""" 4  Can you replace dropout by batch normalization? How does the behavior change?

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.Dropout2d(.2), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Dropout2d(.3), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.Dropout(.4), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Dropout(.5), nn.Sigmoid(),
    nn.Linear(84, 10))
No, the model does not stabilize while replacing batch normalization with dropout under a learning rate of 10.0 """

""" 5  Fix the parameters beta and gamma, and observe and analyze the results.
Performance drops:
loss 1.417, train acc 0.780, test acc 0.606
30345.2 examples/sec on cuda:0
Training loss never drops below 1 -- the network seems to suffer from high bias.

See 7_5_batch_normalization_scratch.py from scratch file for implementation.
"""

""" 6  Review the online documentation for BatchNorm from the high-level APIs to see the other
    applications for batch normalization.

help(torch.nn.BatchNorm1d)
"""

""" 7  Research ideas: think of other normalization transforms that you can apply?
        Can you apply the probability integral transform?
        How about a full rank covariance estimate?
"""
