"""
If we replace l = loss(output, y) with l = loss(output, y).mean(), we need to change trainer.step(batch_size) to trainer.step(1) for the code to behave identically. Why?

Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules gluon.loss and init. Replace the loss by Huber’s loss.

How do you access the gradient of dense.weight?
"""

# Q1.
"""
If we replace l = loss(output, y) with l = loss(output, y).mean(), we need to change trainer.step(batch_size) to trainer.step(1) for the code to behave identically. Why?

Mean is a reduction operation, making loss into a scalar. Steping expects the loss to be of size
minibatch loss, which we reduced to size [1], therefore we need a stepsize of one.
"""

