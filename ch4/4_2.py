from d2l import mxnet as d2l
from mxnet import gluon, np, npx, autograd
import matplotlib as mpl

mpl.use('MacOSX')
npx.set_np()
batch_size = 256
test_iter, train_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_hiddens))
b2 = np.zeros(num_hiddens)
W3 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

def relu(X):
    return np.maximum(X,0)

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(np.dot(X, W1) + b1)
    H2 = relu(np.dot(H1, W2) + b2)
    return np.dot(H2, W3) + b3

loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 10, 0.1

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
    lambda batch_size: d2l.sgd(params, lr, batch_size) )

mpl.pyplot.show()

# initial results top out around 80% train acc

# d2l.predict_ch3(net, test_iter)
