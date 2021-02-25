from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

# from-scratch implementation
class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, int(num_channels/4),
                               kernel_size=1, padding=0, stride=strides)
        self.conv2 = nn.Conv2d(int(num_channels/4), int(num_channels/4),
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(int(num_channels/4), num_channels,
                               kernel_size=1, padding=0)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(int(num_channels/4))
        self.bn2 = nn.BatchNorm2d(int(num_channels/4))
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        return F.relu(Y + X)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(*resnet_block(64, 256, 3))  # can't get first_block figured out
b3 = nn.Sequential(*resnet_block(256, 512, 4))
b4 = nn.Sequential(*resnet_block(512, 1024, 6))
b5 = nn.Sequential(*resnet_block(1024, 2048, 3))

# ./resnet18.svg.png
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(2048, 10))

# Examine the net
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# RUN
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)

# loss 0.152, train acc 0.943, test acc 0.852
# 642.2 examples/sec on cuda:0

# Resnet50 Conclusions:
# test acc about the same as resnet18.
# train acc ~10 points higher
# loss much lower
