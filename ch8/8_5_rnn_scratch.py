from d2l import torch as d2l
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    """ Specifically, the calculation of the hidden variable of the current
    time step is determined by the input of the current
    time step together with the hidden variable of the previous time step:

    Ht = ϕ(Xt Wxh + Ht−1 Whh + bh)      Eqn. (8.4.5)
    """
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        # H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        H = torch.relu(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        # sample from multinomial instead of argmax
        # outputs.append(int(torch.multinomial(F.softmax(y, dim=1), num_samples=1).reshape(1)))
        # biased α = 2, must be integer
        α = 3
        outputs.append(int(torch.multinomial(F.softmax(y**α, dim=1), num_samples=1).reshape(1)))
        # outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            # l.backward(retain_graph=True)  # Ex.6
            l.backward()
            # grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    plt.show()
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    # predict('heat-ray')
    # predict('falling star')
    print(predict('time traveller'))
    print(predict('traveller'))



# RUN SCRIPT

DEBUG = True
device = d2l.try_gpu()
if DEBUG:
    device = 'cpu'
batch_size, num_steps = 32, 35
num_hiddens = 512
num_epochs, lr = 500, 1

train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

X = d2l.reshape(d2l.arange(10), (2, 5))
net = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# Default Performance Benchmarks:
# perplexity 1.0, 135269.0 tokens/sec on cuda:0

"""
## Exercises

1. Show that one-hot encoding is equivalent to picking a different embedding for
    each object.
2. Adjust the hyperparameters (e.g., number of epochs, number of hidden units,
    number of time steps in a minibatch, and learning rate) to improve the
    perplexity.
    * How low can you go?
    * Replace one-hot encoding with learnable embeddings. Does this lead to
        better performance?
    * How well will it work on other books by H. G. Wells, e.g.,
        [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)?
3. Modify the prediction function such as to use sampling rather than picking
    the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., by sampling from:
        q(xt∣xt−1,…,x1) ∝ P(xt∣xt−1,…,x1)^α for α>1.
4. Run the code in this section without clipping the gradient. What happens?
5. Change sequential partitioning so that it does not separate hidden states
    from the computational graph. Does the running time change? How about the
    perplexity?
6. Replace the activation function used in this section with ReLU and repeat the
    experiments in this section. Do we still need gradient clipping? Why?
"""
"""
1. Obvious
2.1. Skipped
2.2. Referece file 8_5_2_embeddings.py
"""

"""2.3. * How well will it work on other books by H. G. Wells, e.g.,
    [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)?

perplexity 1.0, 127928.7 tokens/sec on cuda:0

from load_war_of_the_worlds import *
train_iter, vocab = load_data_war_of_the_worlds(batch_size, num_steps)
"""

# perplexity 1.0, 104621.2 tokens/sec on cuda:0

""" 3. Modify the prediction function such as to use sampling rather than picking
    the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., by sampling from:
        q(xt∣xt−1,…,x1) ∝ P(xt∣xt−1,…,x1)^α for α>1.

sample from multinomial instead of argmax
outputs.append(int(torch.multinomial(F.softmax(y, dim=1), num_samples=1).reshape(1)))
biased α = 2, must be integer
α = 2
outputs.append(int(torch.multinomial(F.softmax(y**α, dim=1), num_samples=1).reshape(1)))

Good performance either way. Scaling with α leads to a convergence with the
argmax results from the original model.
No scaling leads to more mistakes but also a bit more diversity in output.

argmax seems to work the most consistenly however.
"""

# 4. Run the code in this section without clipping the gradient. What happens?
# Doesn't converge. Just comment out: grad_clipping(net, 1)

# 6. Replace the activation function used in this section with ReLU and repeat the
    # experiments in this section. Do we still need gradient clipping? Why?
    # No. ReLU activations keep the gradient from exploding.

# 5. Change sequential partitioning so that it does not separate hidden states
    # from the computational graph. Does the running time change? How about the
    # perplexity?
# Can't get it to run, even with retain_graph=True on the backward op.
