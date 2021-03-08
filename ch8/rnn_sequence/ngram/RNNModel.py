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

    def predict(self, prefix, num_preds, device):  #@save
        """Generate new characters following the `prefix`."""
        τ = prefix.shape[-1]
        state = self.begin_state(batch_size=1, device=device)
        outputs = torch.zeros(prefix.shape[0], τ + num_preds, device=device)
        prefix = prefix.reshape(-1, 1, τ)
        outputs[:, 0:τ] = prefix[:, 0, :]
        get_input = lambda i: d2l.reshape(outputs[:, i:i+τ], (-1, 1, τ))
        # for y in prefix[1:]:  # Warm-up period
            # _, state = net(get_input(), state)
        for i in range(num_preds):  # Predict `num_preds` steps
            y, state = self.forward(get_input(i), state)
            outputs[:, i+τ] = y.reshape(-1)
        return outputs

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
            # y = Y.T.reshape(-1)
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

    def train(self, net, train_iter, num_preds, lr, num_epochs, device,
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
        predict = lambda prefix: self.predict(prefix, num_preds, net, device)
        # Train and predict
        for epoch in range(num_epochs):
            mse, speed = self.train_epoch(
                net, train_iter, loss, updater, device, use_random_iter)
            if (epoch + 1) % 10 == 0:
                animator.add(epoch + 1, [mse])
        plt.show()
        print(f'mean squared loss {mse:.1f}, {speed:.1f} tokens/sec on {str(device)}')


class Text(nn.Module):
    """The RNN model for text datasets"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(Text, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
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
            # α = 3
            # outputs.append(int(torch.multinomial(F.softmax(y**α, dim=1), num_samples=1).reshape(1)))
            outputs.append(int(y.argmax(dim=1).reshape(1)))
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
