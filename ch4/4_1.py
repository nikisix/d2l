#ACTIVATION FUNCTIONS

from matplotlib import pyplot
import matplotlib as mpl
from sympy.abc import x
from sympy import *


sympy.init_printing(use_unicode=True)
mpl.use('MacOSX')

plot(
    1/(1+exp(-x)),
    exp(-x)/(1+exp(-x))**2
)

sigmoid = 1/(1+exp(-x))

# plot(sigmoid, diff(sigmoid))
plot(sigmoid, diff(sigmoid), diff(diff(sigmoid)))


"""
4.1.4. Exercises
1. Compute the derivative of the pReLU activation function.

2. Show that an MLP using only ReLU (or pReLU) constructs a continuous piecewise linear function.

3. Show that  tanh(𝑥)+1=2sigmoid(2𝑥) .

4. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of problems do you expect this to cause?
"""

# 1. Compute the derivative of the pReLU activation function.
# pReLU(𝑥) = max(0,𝑥)+𝛼min(0,𝑥).

prelu = Max(0,x) + a*Min(0,x)
diff(prelu, x)
# = a*Heaviside(-x) + Heaviside(x)

"""
dx(max(0,x)) = {
        0, x < 0
        1, x > 0
    }
    == Heaviside(x)

dx(min(0,x)) = {
        1, x < 0
        0, x > 0
    }
    == Heaviside(-x)
"""

# 2. Show that an MLP using only ReLU (or pReLU) constructs a continuous piecewise linear function.

"""
Consider an MLP with three layers,
    w1 is the weight between input and hidden,
    w2 being the weight between hidden and ouput:

    input -w1-> hidden -w2-> output

    Applying input x, we end up with:

    (x) -w1-> (x*w1) -> (x*w1*w2)

    an output of: (x*w1*w2)

    Now let''s apply a ReLU activation function and see how the output changes:

    relu(input) -w1-> relu(hidden) -w2-> output

    Again, supply an inupt of x and obtain for output:

    ⎧   0     for x ≤ 0
    ⎨
    ⎩w₁⋅w₂⋅x  otherwise

    This is piecewise linear

Sympy code for this answer:
"""
relu = Piecewise((0, x<=0), (x, x>0))

"""
relu
    ⎧0  for x ≤ 0
    ⎨
    ⎩x  otherwise
"""

w1, w2 = symbols('w1 w2')
Piecewise((0, x<=0), (x*w1*w2, x>0))
"""
    ⎧   0     for x ≤ 0
    ⎨
    ⎩w₁⋅w₂⋅x  otherwise
"""


# 3. Show that  tanh(𝑥)+1 = 2*sigmoid(2𝑥)

class sigmoid(Function):
    @classmethod
    def eval(cls, x):
        return 1/(1+exp(-x))

tanh(x)

th = (1-exp(-2*x))/(1+exp(-2*x))
sigmoid = 1/(1+exp(-x))

lhs = th + 1
rhs = 2/(1+exp(-2*x))

"""
[ins] In [126]: lhs
Out[126]:
     -2⋅x
1 - ℯ
───────── + 1
     -2⋅x
1 + ℯ

[ins] In [127]: rhs
Out[127]:
    2
─────────
     -2⋅x
1 + ℯ
"""
denom = (1+exp(-2*x))
"""
[ins] In [144]: rhs*denom
Out[144]: 2

[nav] In [142]: lhs*denom
           ...:
Out[142]:
            ⎛     -2⋅x    ⎞
⎛     -2⋅x⎞ ⎜1 - ℯ        ⎟
⎝1 + ℯ    ⎠⋅⎜───────── + 1⎟
            ⎜     -2⋅x    ⎟
            ⎝1 + ℯ        ⎠


1-exp(-2*x) + denom
out: 2

simplify(lhs*denom)
Out[143]: 2
"""

# Or just:
simplify(lhs/rhs)
# 1

