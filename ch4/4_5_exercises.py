"""4.5.6. Exercises

1. Experiment with the value of λ in the estimation problem in this section.
    Plot training and test accuracy as a function of λ.
    What do you observe?  Use a validation set to find the optimal value of λ.
    Is it really the optimal value? Does this matter?

2. What would the update equations look like if instead of ∥w∥_2 we used ∑_i|w_i| as our penalty of choice (L1
        regularization)?

3. We know that ∥w∥2=w⊤w . Can you find a similar equation for matrices (see the Frobenius norm in Section 2.3.10)?

4. Review the relationship between training error and generalization error. In addition to weight decay, increased
training, and the use of a model of suitable complexity, what other ways can you think of to deal with overfitting?

    ans: more data, dropout? pooling?

5. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via P(w∣x) ∝ P(x∣w)P(w) .
How can you identify P(w) with regularization?
"""

""" 1. Experiment with the value of λ in the estimation problem in this section.
    Plot training and test accuracy as a function of λ.
    What do you observe?  Use a validation set to find the optimal value of λ.
    Is it really the optimal value? Does this matter?
"""
# from i4_5_weight_decay_concise import *

from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt

"""
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
num_epochs = 5

def train(λ, animator):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    lr = 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': λ},
        {"params":net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        training_loss = d2l.evaluate_loss(net, train_iter, loss)
        test_loss = d2l.evaluate_loss(net, test_iter, loss)
    animator.add(λ, (training_loss, test_loss))


animator = d2l.Animator(xlabel='lambda', ylabel='loss', yscale='log',
                        xlim=[5, num_epochs], legend=['train', 'test'])
for λ in torch.range(.1, 2, .3):
    train(float(λ), animator)

for λ in torch.range(2, 20, 5):
    train(float(λ), animator)

plt.show()
"""
# ----------------------------------------------------------------------------
# 1.1.2
    # What do you observe?  Use a validation set to find the optimal value of λ.
    # Is it really the optimal value? Does this matter?
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
n_eval = 20
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
eval_data = d2l.synthetic_data(true_w, true_b, n_eval)
eval_iter = d2l.load_array(eval_data, batch_size, is_train=False)
eval_epochs, num_epochs = 5, 10

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    λ = torch.zeros(1, requires_grad=True)
    return [w, b, λ]

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_and_eval():
    w, b, λ = init_params()
    loss = nn.MSELoss()
    lr = 0.03
    net = lambda X: torch.matmul(X, w) + b

    # for eval_epoch in range(eval_epochs):
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                l = (y - (torch.matmul(X,w) + b))**2 + λ*torch.sum(w.pow(2))/2
                l.sum().backward()
                sgd([w, b], lr, batch_size)  # works -- sort of
                # sgd([w, b, λ], lr, batch_size)  # all nan losses
        training_loss = d2l.evaluate_loss(net, train_iter, loss)
        test_loss = d2l.evaluate_loss(net, test_iter, loss)
        # eval_loss = d2l.evaluate_loss(net, eval_iter, loss)
        # print(training_loss, test_loss, eval_loss, λ)
        print(training_loss, test_loss, λ)
        animator.add(epoch, (training_loss, test_loss))

        # with torch.enable_grad():
            # eval_loss = d2l.evaluate_loss(net, eval_iter, loss)  # float but need tensor
        # eval_loss.backward()
        # with torch.no_grad():
            # λ -= lr * λ.grad
            # λ.grad.zero_()

animator = d2l.Animator(xlabel='lambda', ylabel='loss', yscale='log',
                        xlim=[5, num_epochs], legend=['train', 'test'])
train_and_eval()

"""
5. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via P(w∣x) ∝ P(x∣w)P(w) .
How can you identify P(w) with regularization?

FROM:
https://en.wikipedia.org/wiki/Regularization_(mathematics)#cite_note-4
A theoretical justification for regularization is that it attempts to impose Occam's razor on the solution (as depicted
in the figure above, where the green function, the simpler one, may be preferred). From a Bayesian point of view, many
regularization techniques correspond to imposing certain prior distributions on model parameters.

FROM:
https://en.wikipedia.org/wiki/Tikhonov_regularization#Bayesian_interpretation
Statistically, the prior probability distribution of x {\displaystyle x} x is sometimes taken to be a multivariate
normal distribution. For simplicity here, the following assumptions are made: the means are zero; their components are
independent; the components have the same standard deviation σ x {\displaystyle \sigma _{x}} \sigma _{x}. The data are
also subject to errors, and the errors in b {\displaystyle b} b are also assumed to be independent with zero mean and
standard deviation σ b {\displaystyle \sigma _{b}} \sigma _{b}. Under these assumptions the Tikhonov-regularized
solution is the most probable solution given the data and the a priori distribution of x {\displaystyle x} x, according
to Bayes' theorem
"""
