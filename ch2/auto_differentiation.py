from mxnet import autograd, np, npx
npx.set_np()

"""
great primer on matrix calculus theory
https://explained.ai/matrix-calculus/

*Vector sum reduction*
"""


""" Before we even calculate the gradient of  𝑦  with respect to  𝐱 , we will need a place to store it. It is important that we do not allocate new memory every time we take a derivative with respect to a parameter because we will often update the same parameters thousands or millions of times and could quickly run out of memory. Note that a gradient of a scalar-valued function with respect to a vector  𝐱  is itself vector-valued and has the same shape as  𝐱 .
"""
x = np.arange(4.0)
x.attach_grad()
x.grad

with autograd.record():
    y = x.sum()
x.grad
y.backward()
with autograd.record():
    y = 2 * np.dot(x,x)

print('x graident', x.grad)
print('y backprop', y.backward())


"""Questions
1. Why is the second derivative much more expensive to compute than the first derivative?
    Ans: chain rule creates more terms

2. After running the function for backpropagation, immediately run it again and see what happens.

    Ans: Error. backprop gradient tape gets deleted upon use without the retain parameter
set to True

3. In the control flow example where we calculate the derivative of d with respect to a, what would happen if we changed the variable a to a random vector or matrix. At this point, the result of the calculation f(a) is no longer a scalar. What happens to the result? How do we analyze this?
    Ans: vector -> matrix

4. Redesign an example of finding the gradient of the control flow. Run and analyze the result.

    Ans: see below

5. Let  𝑓(𝑥)=sin(𝑥) . Plot  𝑓(𝑥)  and  𝑑𝑓(𝑥)𝑑𝑥 , where the latter is computed without exploiting that  𝑓′(𝑥)=cos(𝑥) .
"""

# Q3. CONTROL FLOW EXAMPLE
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


[nav] In [195]: xx = np.array([list(range(3)) for i in range(1, 4)])

"""
[ins] In [196]: xx
Out[196]:
array([[0., 1., 2.],
       [0., 1., 2.],
       [0., 1., 2.]])
"""

xx.attach_grad()

with autograd.record():
    y = f(xx)

y.backward()

"""
[ins] In [201]: xx.grad
Out[201]:
array([[512., 512., 512.],
       [512., 512., 512.],
       [512., 512., 512.]])

[ins] In [200]: y
Out[200]:
array([[   0.,  512., 1024.],
       [   0.,  512., 1024.],
       [   0.,  512., 1024.]])
"""




from sympy import *

f = symbols('f', cls=Function)
x = symbols('x')
f = sin
plot(diff(f(x)), backend='text')

"""
      1 |          .                .                .
        |         . .              . .              . .
        |
        |        .   .            .   .            .   .
        |
        |             .                           .
        |       .                .     .                .
        |
        |
        |      .       .        .       .        .       .
      0 |-------------------------------------------------------
        |
        |
        |     .         .      .         .      .         .
        |
        |
        |    .           .    .           .    .           .
        |
        |.                .                   .                .
        |   .                .             .                .
     -1 |_______________________________________________________
         -10                        0                          10
Out[1]: <sympy.plotting.plot.Plot at 0x110264dc0>

[nav] In [2]: plot(f(x), backend='text')
      1 |              ..               ..               ..
        |             .                .                .
        |                .                .                .
        |            .                .                .
        |
        |.                .                .                .
        |                            .                .
        |           .
        | .                .                .                .
        |
      0 |----------.----------------.----------------.----------
        |
        |  .                .                .                .
        |                                           .
        |         .                .
        |   .                .                .                .
        |
        |        .                .                .
        |    .                .                .
        |       .                .                .
     -1 |_______________________________________________________
         -10                        0                          10
"""
