from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

"""
Stock LeNet Performance:
    loss 0.467, train acc 0.824, test acc 0.797
    48977.1 examples/sec on cuda:0

2. Try to construct a more complex network based on LeNet to improve its accuracy.
    Adjust the convolution window size.
        first conv 5->7. Padding 2->3.
            loss 0.462, train acc 0.826, test acc 0.799
            49152.7 examples/sec on cuda:0

        first and second conv2d 5 -> 7. padding = 3, 1
            loss 0.426, train acc 0.843, test acc 0.799
            44560.2 examples/sec on cuda:0

        conv -> 9. padding = 4, 2
            loss 0.430, train acc 0.841, test acc 0.819
            31436.1 examples/sec on cuda:0

    Adjust the number of output channels.
        Doubled the number of output channels.
        Only marginal improvement in test acc and perf goes way down.

        net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 12, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(12, 24, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(24 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

        loss 0.461, train acc 0.826, test acc 0.801
        29653.8 examples/sec on cuda:0

    Adjust the activation function (e.g., ReLU).

        net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.Sigmoid(),  # Leave this Sigmoid to output categorical probabilities
            nn.Linear(84, 10))

        loss 0.281, train acc 0.893, test acc 0.864
        48312.6 examples/sec on cuda:0
        Huge increase!!

    Adjust the number of convolution layers.
        Adding two extra conv2d (with no extra pooling) layers dumpsters accuracy

        net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.Conv2d(6, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.Conv2d(6, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

        loss 2.304, train acc 0.098, test acc 0.100
        17302.6 examples/sec on cuda:0

        increasing the num_epochs from 10 -> 20 reduced test error:
        (more epochs would certainly improve accuracy substantially)
            loss 0.470, train acc 0.823, test acc 0.820
            39601.1 examples/sec on cuda:0

    Adjust the number of fully connected layers.
        no increase in test acc over baseline
        num_epochs=20
        net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 400), nn.Sigmoid(),
            nn.Linear(400, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))
        loss 0.520, train acc 0.802, test acc 0.801
        45579.8 examples/sec on cuda:0

    Adjust the learning rates and other training details
        (e.g., initialization and number of epochs.)

    * Simplify
        How well can we do with a simpler architecture?
        single channel all the way through
            loss 0.500, train acc 0.814, test acc 0.779
            73743.1 examples/sec on cuda:0

        relu and max pool
            net = torch.nn.Sequential(
                Reshape(),
                nn.Conv2d(1, 1, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(1, 1, kernel_size=5), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(1 * 5 * 5, 25), nn.Sigmoid(),
                nn.Linear(25, 10))
            loss 0.574, train acc 0.789, test acc 0.769
            73634.9 examples/sec on cuda:0

    * Best accuracy
        net = torch.nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

        loss 0.203, train acc 0.923, test acc 0.897
        50960.2 examples/sec on cuda:0
"""

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

""" Original LeNet
net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
"""
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # Required for BERT Fine-tuning (to be covered later)
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    plt.show()


lr, num_epochs = 0.9, 20
train_ch6(net, train_iter, test_iter, num_epochs, lr)


# Exercise examine weights for maximal activation
# Display the activations of the first and second layer of LeNet for different inputs (e.g., sweaters and coats).
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

"""
    0 t-shirt
    1 trouser
    2 pullover
    3 dress
    4 coat
    5 sandal
    6 shirt
    7 sneaker
    8 bag
    9 ankle boot
"""
"""
    for layer in net:
        x0 = layer(x0)
        print(layer.__class__.__name__,'output shape: \t', x0.shape)
        if layer.__class__.__name__ == 'Conv2d':
            conv2d_out = x0
            break
"""

# d2l.show_images(x0.reshape((1, 28, 28)).cpu().detach().numpy(), 3, 2)
# plt.show()

# for i in range(len(output)): print(i, output[i].shape)
"""
0 torch.Size([1, 1, 28, 28]) -> Reshape(output[0]) => output[1]. net[0] == Reshape
1 torch.Size([1, 1, 28, 28]) -> Conv2d(output[1])  => output[2]. ...etc
2 torch.Size([1, 6, 28, 28])
3 torch.Size([1, 6, 28, 28])
4 torch.Size([1, 6, 14, 14])
5 torch.Size([1, 16, 10, 10])
6 torch.Size([1, 16, 10, 10])
7 torch.Size([1, 16, 5, 5])
8 torch.Size([1, 400])
9 torch.Size([1, 120])
10 torch.Size([1, 120])
11 torch.Size([1, 84])
12 torch.Size([1, 84])
13 torch.Size([1, 10])
"""

# net
""" Sequential(
  (0): Reshape()
  (1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (2): Sigmoid()
  (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (4): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (5): Sigmoid()
  (6): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (7): Flatten(start_dim=1, end_dim=-1)
  (8): Linear(in_features=400, out_features=120, bias=True)
  (9): Sigmoid()
  (10): Linear(in_features=120, out_features=84, bias=True)
  (11): Sigmoid()
  (12): Linear(in_features=84, out_features=10, bias=True)
) """
# plotting commands

def plot_activations(net, x, y_label, layer_plot_sizes):
    """Plots forward activations in a neural network
    Params
        net (torch.nn): neural network
        x (tensor): single data point of 4 dimensions: [batch, channel, height, width]
        y_label (string): used for plotting and file_names
        layer_plot_sizes (dict: {layer_num: [(layer_input_shape): (subplots)]
    Output:
        images in directory "activation-imgs/{y_label}/...". You must create this directory
    """
    output = dict()
    output[0] = x
    for i in range(len(net)):
        output[i+1] = net[i](output[i])

    for i in range(len(output)):
        d2l.show_images(
                output[i].reshape(layer_plot_sizes[i][0]).cpu().detach().numpy(),
                num_rows=layer_plot_sizes[i][1][0],
                num_cols=layer_plot_sizes[i][1][1]
            )
        if i == 13: # net only has 12 layers
            plt.savefig(
                f'activation-imgs/{y_label}/{y_label}-layer-{i}-{net[i-1].__class__.__name__}-{layer_plot_sizes[i][0]}')
        else:
            plt.savefig(
                f'activation-imgs/{y_label}/{y_label}-layer-{i}-{net[i].__class__.__name__}-{layer_plot_sizes[i][0]}')



# # Load up first batch of training examples
# for X, y in train_iter:
    # x, y1 = X, y
    # break

# deref first training label
# y1.shape  # 256, 1
# print(text_labels[int(y1[0])])

# img_size, subplots
layer_plot_sizes = {
        0: [(1,28,28), (1,2)],
        1: [(1, 28, 28), (1,2)],
        2: [(6, 28, 28), (3,2)],
        3: [(6, 28, 28), (3,2)],
        4: [(6, 14, 14), (3,2)],
        5: [(16, 10, 10), (4,4)],
        6: [(16, 10, 10), (4,4)],
        7: [(16, 5, 5), (4,4)],
        8: [(1, 20, 20), (1,2)],
        9: [(1, 12, 10), (1,2)],
        10:[(1, 12, 10), (1,2)],
        11:[(1, 12, 7), (1,2)],
        12:[(1, 12, 7), (1,2)],
        13:[(1, 10, 1), (1,2)],
    }

# image_num=14
# device = next(iter(net.parameters())).device
# x0 = x[image_num].reshape(1,1,28,28).to(device)
# y_label = text_labels[int(y1[image_num])].strip()
# plot_activations(net, x0, y_label, layer_plot_sizes)

# some y_labels to plot:
# [(i, text_labels[int(y1[i])]) for i in range(20)]

"""
[(0, 'coat'),
 (1, 'pullover'),
 (2, 'coat'),
 (3, 'shirt'),
 (4, 'shirt'),
 (5, 'ankle boot'),
 (6, 'bag'),
 (7, 'trouser'),
 (8, 'sneaker'),
 (9, 'pullover'),
 (10, 'sneaker'),
 (11, 'sneaker'),
 (12, 'ankle boot'),
 (13, 'sneaker'),
 (14, 'trouser'),
 (15, 'trouser'),
 (16, 'shirt'),
 (17, 't-shirt'),
 (18, 'coat'),
 (19, 'ankle boot')] """


import os
import imageio
def dir_to_gif(dir_name, gif_name):
    # CREATE GIFS FROM IMAGE OUTPUTS
    images = []
    filenames = os.listdir(dir_name)
    for filename in filenames:
        images.append(imageio.imread(f'{dir_name}/{filename}'))
    imageio.mimsave(f'{gif_name}.gif', images)

# dir_to_gif('activation-imgs/sneaker/', 'sneaker-activations')

"""
Exercises (inline)
    1. Replace the average pooling with max pooling. What happens?
        nn.AvgPool2d(kernel_size=2, stride=2), -> nn.MaxPool2d(kernel_size=2, stride=2),
        See: ./lenet-maxpool-both-performance.png . Test accuracy increases.
        Stock Performance:
            loss 0.467, train acc 0.824, test acc 0.797
            48977.1 examples/sec on cuda:0
        Only the second layer was changed from avg -> max:
            loss 0.423, train acc 0.843, test acc 0.819
            47349.2 examples/sec on cuda:0
        Both changed from avg -> max performance:
            loss 0.411, train acc 0.849, test acc 0.841
            50034.5 examples/sec on cuda:0

    2. Try to construct a more complex network based on LeNet to improve its accuracy.
        Adjust the convolution window size.
        Adjust the number of output channels.
        Adjust the activation function (e.g., ReLU).
        Adjust the number of convolution layers.
        Adjust the number of fully connected layers.
        Adjust the learning rates and other training details (e.g., initialization and number of epochs.)
    3. Try out the improved network on the original MNIST dataset.
    4. Display the activations of the first and second layer of LeNet for different inputs (e.g., sweaters and coats).
"""
