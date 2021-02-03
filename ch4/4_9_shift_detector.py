"""COVARIATE SHIFT DETECTOR
Training data sampled from distribution p
Test data sampled from distribution q

We can reuse the training data here for the dual purpose of training the class balance classifier
(accessing test data) as well as training the final model parameters
if covariate shift is absent, then the classifier should have poor accuracy

classifier modeled after 3.2
"""

from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()


def synthetic_data_dist(
        mean1=0, sd1=1, label1=0,
        mean2=1, sd2=1, label2=1,
        num_examples=100,
        num_params=1,
        class_balance=.5):  #@save
    """Draw samples from two normal distributions
    class_balance: 0 < prob < 1 of drawing from class 1 (else draw class2)"""
    num_ex1 = int(np.round(num_examples*class_balance))
    num_ex2 = int(np.round(num_examples*(1-class_balance)))
    x1 = np.random.normal(mean1, sd1, (num_ex1, num_params))
    y1 = np.ones((num_ex1, 1))*label1
    x2 = np.random.normal(mean2, sd2, (num_ex2, num_params))
    y2 = np.ones((num_ex2, 1))*label2
    X = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    # shuffle indices
    ix = list(range(num_examples))
    random.shuffle(ix)
    return X[ix], y[ix]

# READ DATASET
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# MODEL
def logreg(X, w, b):  #@save
    """The logistic regression model."""
    return 1/(1 + np.exp(np.dot(X, w.T))) + b

# LOSS FUNCTION
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape))**2)/2

# OPTIMIZATION ALGORITHM
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - (lr * param.grad / batch_size)

# TRAINING
def train(features, labels, w, b, epochs, lr, batch_size, loss, net):
    for epoch in range(epochs):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y)
            l.backward()  # Compute gradients
            sgd([w, b], lr, batch_size)  # Update parameters using their gradient
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    return [w,b]


# INIT DATA, PARAMS AND TRAIN MODEL
mean1, sd1 = 0, 1
mean2, sd2 = 100, 20
n_train, n_test, num_examples, batch_size, num_params = 80, 20, 100, 5, 2
train_data, train_labels = synthetic_data_dist(
        mean1=mean1, sd1=sd1, mean2=mean2, sd2=sd2,
        num_params=num_params, num_examples=n_train, class_balance=.33)
# train_iter = data_iter(batch_size, features=train_data, labels=train_labels)

# test_data, test_labels = synthetic_data_dist(num_params=num_params, num_examples=n_test, label=1)
# test_iter = data_iter(batch_size, features=test_data, labels=test_labels)

# INITILZE PARAMETERS (w)
w = np.random.normal(0, 1, size=(1, num_params))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

w_hat, b_hat = train(
      train_data
    , train_labels
    , w
    , b
    , epochs=10
    , lr=.2
    , batch_size=batch_size
    , loss=squared_loss
    , net=logreg)

print("params after training:", w_hat, b_hat)

y_hat = logreg(train_data, w_hat, b_hat)
p_q = y_hat.mean() # estimated class balance ratio


# if logreg(x, w, b) is from class one then:
# l = p_q*loss(net(X, w, b), y)
# else:
# l = (1 - p_q)*loss(net(X, w, b), y)
