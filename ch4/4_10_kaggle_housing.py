from d2l import torch as d2l
from functools import partial
from torch import nn
import numpy as np
import pandas as pd
import torch

from data_download import download, DATA_URL as DATA_URL, DATA_HUB as DATA_HUB

# For convenience, we can download and cache the Kaggle housing dataset using 
# the script we defined above.
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

## PANDAS PREPROCESSING ################
# Combine train and test, while removing native Id field
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# print(train_data.shape)
# print(test_data.shape)
# print(all_features.shape)

# Normalize numeric columns: x ← (x−μ)/σ.
# TODO fix info leakage here by using mean and sd of train set ONLY
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# Non-numeric columns
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)

# print(all_features.shape)
# Note: columns went from 79->331 from the dummies

## END PREPROCESSING #################

# CONVERT TO TENSOR
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


# Training
""" Log MSE Loss:
           ______________________________
          ╱   n
         ╱   ___
        ╱    ╲
       ╱      ╲                        2
      ╱       ╱   (log(y) - log(y_hat))
     ╱       ╱
    ╱        ‾‾‾
   ╱        i = 1
  ╱         ────────────────────────────
╲╱                       n

Use log loss b/c loss at the high-end of housing market, should carry less
weight than loss at the low-end of the market.
"""
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    # net = nn.Sequential(nn.Linear(in_features,1))
    dropout1, dropout2, dropout3 = .2, .5, .7
    net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU(),
            # nn.Dropout(dropout1),

            nn.Linear(8, 1)
        )
    return net


def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

# People tend to find that Adam is significantly less sensitive to the initial learning rate.
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K-Fold Cross-Validation

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# # In this example, we pick an untuned set of hyperparameters and leave it up to the reader to improve the model.
# k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

from itertools import chain

params = dict()
calls = 0
def bsearch(f_1, s, d):
    global calls
    global params
    calls += 1

    x=list()
    for i in range(len(s)):
        x.append(int((s[i].stop + s[i].start)/2))
    _, fx = fn(learning_rate=x[0], weight_decay=x[1])  # cache money
    # print(x, fx)
    params[fx] = list(x)
    print(fx, f_1, '\t', x)
    if f_1 < fx: return
    if any([s[i].stop - s[i].start <= epsilon for i in range(len(s))]): 
        return fx

    s_left, s_right = s.copy(), s.copy()
    r_bound = [x[d]+1, s[d].stop][bool(x[d]+1>s[d].stop)]
    l_bound = [x[d]-1, s[d].start][bool(x[d]-1 < s[d].start)]
    s_right[d] = slice(r_bound, s[d].stop, 1)
    s_left[d] = slice(s[d].start, l_bound, 1)

    return min(filter(lambda o: o!=None,
        chain( 
            [fx],
            [bsearch(fx, s_left, i) for i in range(len(s)) if i!=d],
            [bsearch(fx, s_right, i) for i in range(len(s)) if i!=d],
        )
    ))

epsilon = 1
k, num_epochs = 2, 10
batch_size = 256
# batch_size = slice(32, 256, 1)
# lr, weight_decay, batch_size = 5, 0, 64
lr = slice(0, 100, 1)
weight_decay = slice(0, 5, 1)
fn = partial(k_fold, k, train_features, train_labels, num_epochs, batch_size=batch_size)
fx_init = fn(learning_rate=lr.start, weight_decay=weight_decay.stop)[1]
print(fx_init, 'fx_init')
print(bsearch(fx_init, [lr, weight_decay], 0))
print(params[min(params)], 'minimizes f over the range')
print(calls, 'function calls')

#Plugin optimal params
lr_hat, wd_hat = params[min(params)]
batch_size_hat = 256
k_hat= 5
num_epochs_hat = 100
train_l, valid_l = k_fold(
        k_hat, train_features, train_labels,
        num_epochs=num_epochs_hat, learning_rate=lr_hat,
        weight_decay=wd_hat, batch_size=batch_size_hat)

print(train_l, valid_l, 'final score')

# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      # f'avg valid log rmse: {float(valid_l):f}')

# batch_size = TODO don't support discrete search variables yet


# λ^ = 0
# lr^= [67.53125], 0.17495790123939514 [67.53125] minimizes f over the range 127 function calls
"""
minimize batch_size based on exectution time rather than model accuracy:
    t1 = d2l.time.perf_counter()
    _, fx = fn(batch_size=x[0])  # cache money
    t2 = d2l.time.perf_counter()
    fx = t2-t1
"""
# batch_size^ = 256

# Plug in optimal values
# k_fold(5, train_features, train_labels, num_epochs=100, learning_rate=67, weight_decay=0, batch_size=256)
# Out[289]: (0.13091025352478028, 0.1498199850320816)

# k_fold(5, train_features, train_labels, num_epochs=100, learning_rate=24, weight_decay=2, batch_size=256)
# Out[301]: (7.934028816223145, 8.044289779663085)


################################################################################
# Save predictions for kaggle submission
################################################################################
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls],
             xlabel='epoch',
             ylabel='log rmse',
             xlim=[1, num_epochs],
             yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = net(test_features).detach().numpy()
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


# num_epochs, lr, weight_decay, batch_size = 200, 24, 2, 256
# train_and_pred(train_features, test_features, train_labels, test_data,
               # num_epochs, lr, weight_decay, batch_size)
