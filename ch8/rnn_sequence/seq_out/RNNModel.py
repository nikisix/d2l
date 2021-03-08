from d2l import torch as d2l
from torch import functional as F
from torch import nn
from torch import tensor
import math
import matplotlib.pyplot as plt
import torch


class Numeric(nn.Module):
    """The RNN model for numeric datasets"""
    def __init__(self, rnn_layer, output_size, **kwargs):
        super(Numeric, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.input_size  = self.rnn.input_size
        self.output_size = output_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.output_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

    def forward(self, X, state):
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `output_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

    def grad_clipping(self, net, theta):  #@save
        """Clip the gradient."""
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    def predict(self, prefix, prediction_seq, device):  #@save
        """Generate new characters following the `prefix`."""
        state = self.begin_state(batch_size=1, device=device)
        for y in prefix:  # Warm-up period
            y_hat, state = self.forward(y.reshape(-1, 1, *y.shape), state)
        y_hat, _ = self.forward(prediction_seq.reshape(-1, 1, *prediction_seq.shape), state)
        return y_hat

    def train_epoch(self, net, train_iter, loss, updater, device, use_random_iter):
        """Train a net within one epoch (defined in Chapter 8)."""
        state, timer = None, d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        i = 0
        for X, Y in train_iter:
            X = X.to(torch.float32)  # TODO is this necessary?
            Y = Y.to(torch.float32)
            X = X.reshape(-1, *X.shape)  # [direction, batch_size, seq_len]
            if state is None or use_random_iter:
                # Initialize `state` when either it is the first iteration or
                # using random sampling
                state = net.begin_state(batch_size=X.shape[1], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # `state` is a tensor for `nn.GRU`
                    state.detach_()
                else:
                    # `state` is a tuple of tensors for `nn.LSTM` and
                    # for our custom scratch implementation 
                    for s in state:
                        s.detach_()
            X, Y = X.to(device), Y.to(device)
            y_hat, state = net(X, state)
            l = loss(y_hat, Y).mean()
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.backward()
                self.grad_clipping(net, 1)
                updater.step()
            else:
                l.backward()
                self.grad_clipping(net, 1)
                # Since the `mean` function has been invoked
                updater(batch_size=1)
            metric.add(l * d2l.size(Y), d2l.size(Y))
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    def train(self, net, train_iter, lr, num_epochs, device,
                  use_random_iter=False):
        """Train a model (defined in Chapter 8)."""
        loss = nn.MSELoss()
        animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                                legend=['train'], xlim=[10, num_epochs])
        # Initialize
        if isinstance(net, nn.Module):
            updater = torch.optim.SGD(net.parameters(), lr)
        else:
            updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
        # Train and predict
        for epoch in range(num_epochs):
            mse, speed = self.train_epoch(
                net, train_iter, loss, updater, device, use_random_iter)
            if (epoch + 1) % 10 == 0:
                animator.add(epoch + 1, [mse])
        # plt.show()
        print(f'mean squared loss {mse:.1f}, {speed:.1f} tokens/sec on {str(device)}')
