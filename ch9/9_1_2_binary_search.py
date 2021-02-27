""" Find optimal hyperparams for the GRU net via bianary search"""

from bsearch import BinarySearch
from d2l import torch as d2l
from functools import partial
from torch import nn
from m9_1_2_custom_train_ch8 import train_ch8_slim

# Initial Hypers
batch_size, num_steps = 32, 35
num_epochs, lr = 10, 1
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_inputs = vocab_size

# Optmized Hypers
# num_hiddens = slice(128, 512, 1)
lr = slice(1, 10, .1)

def init_run_gru(
        train_iter, num_epochs, batch_size, num_inputs, num_hiddens, lr):
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    return train_ch8_slim(model, train_iter, vocab, lr, num_epochs, device)


fn = partial(init_run_gru, train_iter, num_epochs, batch_size, num_inputs,
        num_hiddens, lr=lr)

fn_init = fn(lr=lr.start)
bsearch = BinarySearch(fn, epsilon=.1, debug=True)

bsearch(fn_init, [lr], 0)

print("binary search results")
print(bsearch.params)
print('num calls', bsearch.calls)
