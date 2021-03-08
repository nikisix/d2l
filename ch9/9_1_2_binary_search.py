""" Find optimal hyperparams for the GRU net via bianary search"""

from bsearch import BinarySearch
from d2l import torch as d2l
from functools import partial
from torch import nn
from m9_1_2_custom_train_ch8 import train_ch8_slim


def init_run_gru(
        num_epochs, batch_size, num_steps, num_hiddens, lr):
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, device = len(vocab), d2l.try_gpu()
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    return train_ch8_slim(model, train_iter, vocab, lr, num_epochs, device)

# Initial Hypers
batch_size, num_steps = 32, 35
num_hiddens = 256
num_epochs, lr = 10, 1

# Optmize Hypers
batch_size = slice(10, 40)
num_steps = slice(10, 40)
num_hiddens = slice(64, 512, 1)
lr = slice(0, 20, .1)
opt_params = [batch_size, num_steps, num_hiddens, lr]

fn = partial(init_run_gru, num_epochs)
fn_init = fn(*[param.start for param in opt_params])
bsearch = BinarySearch(fn, opt_params, debug=True)
bsearch(fn_init, opt_params, 0)

print("binary search results")
best_score, worst_score = min(bsearch.params), max(bsearch.params)
print('optimal params:', bsearch.params[min(bsearch.params)])
print('worst - best (lower is better):', worst_score - best_score)
print('num calls', bsearch.calls)

""" Hyper Opts Results
varying num_hiddens doesn't affect things too much. 128 ok
lr_hat: 4.5
batch_size_hat, num_steps_hat = [33, 33]

num_steps
Lower is better for perplexity, but MUCH slower

batch_size
Lower is better for perplexity, but MUCH slower
perplexity 5.7, 11491.2 tokens/sec on cuda:0
perplexity 13.1, 222357.8 tokens/sec on cuda:0
13.145983938938462 5.723337618944193     [20]
perplexity 10.5, 115908.5 tokens/sec on cuda:0
10.539268496586832 13.145983938938462    [10]
perplexity 9.2, 59770.1 tokens/sec on cuda:0
9.170914290051996 10.539268496586832     [5]
perplexity 7.3, 23379.3 tokens/sec on cuda:0
7.318990272695167 9.170914290051996      [2]
perplexity 5.7, 11425.9 tokens/sec on cuda:0
5.668787661124166 7.318990272695167      [1]
perplexity 8.2, 36538.2 tokens/sec on cuda:0
8.239951670630207 7.318990272695167      [3]
perplexity 9.9, 82264.3 tokens/sec on cuda:0
9.914420530930226 9.170914290051996      [7]
perplexity 11.6, 172806.5 tokens/sec on cuda:0
11.603040567181862 10.539268496586832    [15]
perplexity 15.2, 307564.7 tokens/sec on cuda:0
15.17869633955476 13.145983938938462     [30]
binary search results
optimal params: [1]
worst - best (lower is better): 9.509908678430595
num calls 9


FULL PARAM SEARCH
Actually just the tail of the search b/c the power saver put the computer to
sleep
5.363763970787123 5.653761960811127      [11, 20, 484, 7.25]
perplexity 5.9, 57538.7 tokens/sec on cuda:0
5.8789398780925515 6.781664112269181     [11, 19, 484, 7.25]
perplexity 5.5, 57337.2 tokens/sec on cuda:0
5.516012509335964 5.8789398780925515     [11, 19, 484, 5.875]
perplexity 5.8, 52286.8 tokens/sec on cuda:0
5.802522685932811 5.516012509335964      [10, 19, 484, 5.875]
perplexity 5.2, 51888.1 tokens/sec on cuda:0
5.167846531548747 5.516012509335964      [10, 19, 484, 5.875]
perplexity 5.6, 52195.0 tokens/sec on cuda:0
5.628781692929749 5.516012509335964      [10, 19, 484, 5.875]
perplexity 5.9, 61269.5 tokens/sec on cuda:0
5.910145380898651 5.516012509335964      [12, 19, 484, 5.875]
perplexity 5.7, 61734.7 tokens/sec on cuda:0
5.666697411868684 5.516012509335964      [12, 19, 484, 5.875]
perplexity 5.7, 61067.1 tokens/sec on cuda:0
5.691813840578852 5.516012509335964      [12, 19, 484, 5.875]
perplexity 5.8, 56885.7 tokens/sec on cuda:0
5.781573756150058 5.8789398780925515     [11, 19, 484, 5.875]
perplexity 5.4, 55822.7 tokens/sec on cuda:0
5.390295609246207 5.781573756150058      [11, 18, 484, 5.875]
perplexity 5.4, 56234.5 tokens/sec on cuda:0
5.390769768718682 5.781573756150058      [11, 18, 484, 5.875]
perplexity 5.5, 52473.5 tokens/sec on cuda:0
5.549756093166835 5.781573756150058      [11, 18, 484, 5.875]
"""

