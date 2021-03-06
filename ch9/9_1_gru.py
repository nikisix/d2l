from d2l import torch as d2l
import torch
from torch import nn

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    # W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


# Hyperparameters
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
print('scratch model')
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
print('concise model')
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

"""
outputs and error look good
# params = [W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_hq, b_q]
def gru_no_reset(inputs, state, params):
    W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        H_tilda = torch.tanh((X @ W_xh) + (H @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

time traveller for so it will be convenient to speak of himwas e
<Figure size 350x250 with 1 Axes>
time traveller tore ansthrspade he hay dearonillespsof chisel a
<Figure size 350x250 with 1 Axes>
time traveller with a slight accession ofcheerfulness really thi
<Figure size 350x250 with 1 Axes>
perplexity 1.0, 58104.0 tokens/sec on cpu
time traveller with a slight accession ofcheerfulness really thi
traveller with a slight accession ofcheerfulness really thi
"""

""" 2. Adjust the hyperparameters and analyze the their influence on running time, perplexity, and the output sequence.
ref: 9_1_binary_search.py
batch size and num steps main affect running time. when they go down running 
time slows down, but perplexity does come down. but not enough to justify 
step/batch sizes of 1.

lr_hat: 4.5
batch_size_hat, num_steps_hat = [33, 33]
interestingly, varying num_hiddens didn't affect things too much.
"""

""" ## Exercises
1. Assume that we only want to use the input at time step t' to predict the output at time step t > t'.
    What are the best values for the reset and update gates for each time step?
2. Adjust the hyperparameters and analyze the their influence on running time, perplexity, and the output sequence.
3. Compare runtime, perplexity, and the output strings for `rnn.RNN` and `rnn.GRU` implementations with each other.
4. What happens if you implement only parts of a GRU, e.g., with only a reset gate or only an update gate? """
"""
1. Assume that we only want to use the input at time step t' to predict the output at time step t > t'.
    What are the best values for the reset and update gates for each time step?

At t == t':
    Reset Gate resets the state when R == 0.
    So set R := 0, to reset the state and allow current input to envelope the state.

At t > t':
    Update Gate is "off" (blocks updates) when Z == 1.
    So set Z = 1, to retain state H_t-1 until the end of time.
"""
