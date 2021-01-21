# see https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_linear-networks/linear-regression-scratch.ipynb

# NOTES
"""
Initialize parameters  (𝐰,𝑏) 

Repeat until done

Compute gradient  𝐠←∂(𝐰,𝑏)1||∑𝑖∈𝑙(𝐱(𝑖),𝑦(𝑖),𝐰,𝑏) 

Update parameters  (𝐰,𝑏)←(𝐰,𝑏)−𝜂𝐠 

"""

"""
3.2.9. Exercises
What would happen if we were to initialize the weights to zero. Would the algorithm still work?
NO

Assume that you are Georg Simon Ohm trying to come up with a model between voltage and current. Can you use auto differentiation to learn the parameters of your model?

Can you use Planck’s Law to determine the temperature of an object using spectral energy density?
YES

What are the problems you might encounter if you wanted to compute the second derivatives? How would you fix them?

Why is the reshape function needed in the squared_loss function?
MINIBATCHING

Experiment using different learning rates to find out how fast the loss function value drops.

If the number of examples cannot be divided by the batch size, what happens to the data_iter function’s behavior?
REDUCES THE SIZE OF THE LAST BATCH.
"""

# Q1. What would happen if we were to initialize the weights to zero. Would the algorithm still work?
# No, vanishing gradient.


"""
Q2. Assume that you are Georg Simon Ohm trying to come up with a model between voltage and current. Can you use auto differentiation to learn the parameters of your model?

Ohm's Law
V=IR, Voltage=Current*Resistance

Problem Set-Up:
Set resistance = 2, observe current, solve for voltage
"""

from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


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


# DEFINE MODEL
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return np.dot(X, w) + b


# DEFINE LOSS FUNCTION
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape))**2)/2


# DEFINE OPTIMIZATION ALGORITHM
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - (lr * param.grad / batch_size)

# TRAINING
"""
Initialize parameters  (𝐰,𝑏)
Repeat until done:
    Compute gradient  𝐠 ← ∂_(𝐰,𝑏) 1/|B| ∑_𝑖∈ B 𝑙oss(𝐱(𝑖),𝑦(𝑖),𝐰,𝑏)
    Update parameters  (𝐰,𝑏) ← (𝐰,𝑏)−𝜂𝐠
    WHERE:
        𝜂 - learning rate
        B - mini batch
        ∂ - partial derivative
"""
def train(
      features
    , lables
    , w
    , b
    , epochs
    , lr
    , batch_size
    , loss
    , net):
    for epoch in range(epochs):
        for X, y_hat in data_iter(batch_size, features, labels):
            with autograd.record():
                y = net(X, w, b)
                l = loss(y_hat, y)
            l.backward()  # Compute gradients
            sgd([w, b], lr, batch_size)  # Update parameters using their gradient
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    return [w,b]


# # INIT DATA, PARAMS AND TRAIN MODEL

# RESISTANCE = 3  # Infer this secret variable

# true_w = np.array([RESISTANCE])
# true_b = 0
# features, labels = synthetic_data(true_w, true_b, 1000)

# # INITILZE PARAMETERS (w)
# w = np.random.normal(0, 1, len(true_w))
# b = np.zeros(1)
# w.attach_grad()
# b.attach_grad()

# w_hat, b_hat = train(
      # features
    # , labels
    # , w
    # , b
    # , epochs=3
    # , lr=.2
    # , batch_size=10
    # , loss=squared_loss
    # , net=linreg)

# print("Underlying resistance inference error:", true_w - (w_hat + b_hat))


# Q3. Can you use Planck’s Law to determine the temperature of an object using spectral energy density?
"""
From section 3.1.3:
"It follows that minimizing the mean squared error is equivalent to maximum likelihood
estimation of a linear model under the assumption of additive Gaussian noise."

Ohm's Law:
B(v,T) = 2v^3/(e^(v/T) - 1)

Is a gaussian in form, so can be interpreted as a linear model with gaussian noise
errors:
Eq. 3.1.15
−log𝑃(𝐲∣𝐗) = ∑𝑖_1_𝑛( 1/2 log(2𝜋𝜎2) + 1/(2 𝜎^2) (𝑦(𝑖) − 𝐰⊤𝐱(𝑖) − 𝑏)^2 )


From Wikipedia
https://en.wikipedia.org/wiki/Planck%27s_law
The spectral radiance can also be expressed per unit wavelength λ instead of per unit frequency. By choosing an appropriate system of unit of measure (i.e. natural Planck units), the law can be simplified to become:

v := Frequency
T := Temperature

Ohm's Law:
B(v,T) = 2v^3/(e^(v/T) - 1)

B := black-body radiation

Solve for Temperature:

B(v,T) = 2v^3/(e^(v/T) - 1)

(e^(v/T) - 1) * B(v,T) = 2v^3

e^(v/T) - 1 = 2v^3/B(v,T)

e^(v/T)= 2v^3/B(v,T) + 1

v/T = log(2v^3/B(v,T) + 1)

T = v/log(2v^3/B(v,T) + 1)
--------------------------

Solve for the P(T|v)

B(v,T) = 2v^3/(e^(v/T) - 1)
"""

# Failed attempt at using a linear model on a negative exponential when we didn't have
# to b/c reduces to a linear model with gaussian measurement error

# N = 1000
# features = np.random.normal(0,1,[N, 2])

# # SANITY CHECK passes
# # labels = 3*features[:,0] + 2*features[:,1]
# labels = (2*features[:,0]**3)/(np.exp(features[:,0]/features[:,1]) - 1)
# # labels = features[:,0]**2)+features[:,1]**2


# # INIT PARAMS
# w = np.random.normal(0, 1, features.shape[1])
# b = np.zeros(1)
# w.attach_grad()
# b.attach_grad()

# """ Indeed it belly flops. Let's try replacing the linear model
    # run1
    # epoch 1, loss 81378.796875
    # epoch 2, loss 7000292.000000
    # epoch 3, loss 603437184.000000

    # run2
    # epoch 1, loss 13677.464844
    # epoch 2, loss 1028122.000000
    # epoch 3, loss 77410736.000000
# """

# # DEFINE MODEL
# def nonlinreg(X, w, b):  #@save
    # """
    # epoch 1, loss 90.331528
    # epoch 2, loss 22.160156
    # epoch 3, loss 25.722517 """
    # # return np.dot(np.exp(-X), w) + b
    # return X[:,0]**w[0] + X[:,1]**w[1] + b
    # # return X*w + b

# # attempt with a linear model, prob not gonna go very well
# w_hat, b_hat = train(
      # features
    # , labels
    # , w
    # , b
    # , epochs=3
    # , lr=.1
    # , batch_size=10
    # , loss=squared_loss
    # , net=nonlinreg)
