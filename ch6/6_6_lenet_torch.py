from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

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


lr, num_epochs = 0.9, 10
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

for X, y in train_iter:
    x, y1 = X, y
    break

# deref first training label
# y1.shape  # 256, 1
print(text_labels[int(y1[0])])

device = next(iter(net.parameters())).device
# deref first training example

"""
    for layer in net:
        x0 = layer(x0)
        print(layer.__class__.__name__,'output shape: \t', x0.shape)
        if layer.__class__.__name__ == 'Conv2d':
            conv2d_out = x0
            break
"""

d2l.show_images(x0.reshape((1, 28, 28)).cpu().detach().numpy(), 3, 2)
plt.show()

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

def plot_activations(x, y_label):
    output = dict()
    output[0] = x
    for i in range(len(net)):
        output[i+1] = net[i](output[i])

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

image_num=14
x0 = x[image_num].reshape(1,1,28,28).to(device)
y_label = text_labels[int(y1[image_num])]
plot_activations(x0, y_label)

# some y_labels to plot:
[(i, text_labels[int(y1[i])]) for i in range(20)]

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
