# sympy examples

"""NOTES
In short, we can think of the cross-entropy classification objective in two ways: (i) as maximizing the likelihood of the observed data; and (ii) as minimizing our surprisal (and thus the number of bits) required to communicate the labels.
"""



"""
3.4.9. Exercises
1. We can explore the connection between exponential families and the softmax in some more depth.

    1.1 Compute the second derivative of the cross-entropy loss  𝑙(𝐲,𝐲̂ )  for the softmax.

    1.2 Compute the variance of the distribution given by  softmax(𝐨)  and show that it matches the second derivative computed above.

2. Assume that we have three classes which occur with equal probability, i.e., the probability vector is  (13,13,13) .

    What is the problem if we try to design a binary code for it?

    Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode  𝑛  observations jointly?

3. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as  RealSoftMax(𝑎,𝑏)=log(exp(𝑎)+exp(𝑏)) .

    Prove that  RealSoftMax(𝑎,𝑏)>max(𝑎,𝑏) .

    Prove that this holds for  𝜆−1RealSoftMax(𝜆𝑎,𝜆𝑏) , provided that  𝜆>0 .

    Show that for  𝜆→ ∞  we have  𝜆−1RealSoftMax(𝜆𝑎,𝜆𝑏)→ max(𝑎,𝑏) .

    What does the soft-min look like?

    Extend this to more than two numbers.
"""

"""
1. We can explore the connection between exponential families and the softmax in some more depth.

    Compute the second derivative of the cross-entropy loss  𝑙(𝐲,𝐲̂)  for the softmax.

note:
    j is contained in k
    k are the classification classes

From the text we have for the loss:
𝑙(𝐲,𝐲̂ ) = log(∑_𝑘 exp(o_k) ) - ∑_j(y_j o_j)

and its first derivative:

∂𝑜_𝑗 𝑙(𝐲,𝐲̂ ) =

    = exp(𝑜_𝑗)/∑_𝑘(exp(𝑜_𝑘)) − 𝑦_𝑗

    = softmax(𝐨)_𝑗 − 𝑦_𝑗.


0. Solve for first derivative ∂𝑜_𝑗 𝑙(𝐲,𝐲̂ ):

    ∂𝑜_𝑗 𝑙(𝐲,𝐲̂ ) = log(∑_𝑘(exp(o_k))) - ∑_j(y_j o_j)

    do_j log(∑_𝑘(e^o_k)) - do_j( ∑_j(y_j o_j) )

    do_j log(∑_𝑘(exp(o_k))) - ∑_j(do_j( y_j o_j ))

    do_j log(∑_𝑘(exp(o_k))) - ∑_j(y_j)

    note:
        do_j ∑_𝑘 exp(o_k) == exp(o_j)

    ∑_𝑘(exp(o_k))^-1 * exp(o_j) - ∑_j(y_j)

    note:
        softmax(o)_j := ∑_𝑘(exp(o_k))^-1 * exp(o_k)
        ∑_j(y_j) =?= y_j

    ∂𝑜_𝑗 𝑙(𝐲,𝐲̂ ) = softmax(o)_j - y_j


1. Solve for the second derivative ∂2𝑜_𝑗
    (or first derivative of the derivative from above):

    ∂𝑜_𝑗( ∑_𝑘(exp(o_k))^-1 * exp(o_j) - y_j )

    ∂𝑜_𝑗 ∑_𝑘(exp(o_k))^-1 * exp(o_j) - ∂𝑜_𝑗 y_j

    ∂𝑜_𝑗 ∑_𝑘(exp(o_k))^-1 * exp(o_j) - 0

    ∂𝑜_𝑗 ∑_𝑘(exp(o_k))^-1 * exp(o_j) - 0

    ∂𝑜_𝑗( e^o_j/∑_𝑘 e^o_k )

    note:
        quotient rule
        dx (N/D) = (DN' - ND')/D'^2, where N and D are functions of x

        and from above
        do_j ∑_𝑘 exp(o_k) == exp(o_j)

    ( ∑_𝑘(e^o_k * e^o_j) - e^o_j * e^o_j ) / e^(o_j*2)

    e^o_j ( ∑_𝑘 e^o_k - e^o_j ) / e^(o_j*2)

    ( ∑_𝑘 e^o_k - e^o_j ) / e^(o_j)

    ∂2𝑜_𝑗 = ( ∑_𝑘 e^o_k - e^o_j ) / e^(o_j)


2. Compute the variance of the distribution given by softmax(𝐨) and show that it matches the second derivative computed above.


var(softmax(o)) =

Recall the expectation is computed as:

    𝐸[𝑋] = ∑_𝑥(𝑥 𝑃(𝑋=𝑥))

Let's start with the expectation of softmax(o):

softmax(o) := exp(o_j)/∑_𝑘(exp(o_k))

note:
    softmax(o) is a distribution:

E[softmax(o)] =  ∑_o o*exp(o)/∑_𝑘(exp(o))


And the variance is:

    Var[𝑋] = 𝐸[(𝑋 − 𝐸[𝑋])^2] = 𝐸[𝑋^2]−𝐸[𝑋]^2

Var[softmax(o)] =  𝐸[softmax(o^2)]−𝐸[softmax(o)]^2

∑_o o^2*exp(o^2)/∑_𝑘(exp(o^2)) - (∑_o o*exp(o)/∑_𝑘(exp(o)))^2

∑_o o^2*exp(o^2)/∑_𝑘(exp(o^2)) - ∑_o o^2*exp(2*o)/∑_𝑘(exp(2*o))

∑_o [ o^2*exp(o^2)/∑_𝑘(exp(o^2)) - o^2*exp(2*o)/∑_𝑘(exp(2*o)) ]

factor out o^2:

∑_o [ o^2 ( exp(o^2)/∑_𝑘(exp(o^2)) - exp(2*o)/∑_𝑘(exp(2*o)) ]
"""


# Take 2 -- attempt in sympy

from sympy import *
from sympy.abc import x,y,z

init_printing(use_unicode=True)

f, g, h = symbols('f g h', cls=Function)
softmax = symbols('softmax', cls=Function)
loss_def = symbols('loss_def', cls=Function)
loss = symbols('loss', cls=Function)
y_h = symbols('y_h', cls=Function, cls=seq)

j, k, q = symbols('x o j k q', integer=True)
# y, y_h = symbols('y y_h')

o = sympy.IndexedBase("o")
y = sympy.IndexedBase("y")

k = Idx("k", (1, q))
j = Idx("j", (1, q))

# EQN 3.4.3 Choice Model Softmax
y_h = exp(o[j])/Sum(exp(o[k]), k)
softmax = exp(o[j])/Sum(exp(o[k]),k)

# Cross Entropy Loss
loss_def = -1*summation(y*log(y_h), (j, 1, q))

loss = -1*summation(y*log(softmax), (j, 1, q))

diff(o[j]/Sum(o, k), o[j])

loss = -Sum(
        y[j]*
        log(
            exp(o[j])/
            Sum(exp(o[k]),k)
    ), j)

# 3.4.9 simplified loss
loss_1 = log( Sum(exp(o[k], k) ) - Sum(y[j]*o[j], j)

diff(loss_1, o[j])
"""
Out[273]:
  q
 ____
 ╲
  ╲
   ╲   o[k]
   ╱  ℯ    ⋅δ
  ╱          j,k     q
 ╱                  ___
 ‾‾‾‾               ╲
k = 0                ╲
──────────────── -   ╱   y[j]
    q               ╱
   ___              ‾‾‾
   ╲               j = 0
    ╲    o[k]
    ╱   ℯ
   ╱
   ‾‾‾
  k = 0


simplifies to:

     o[j]
    ℯ
────────────────  -  y[j]
    q
   ___
   ╲
    ╲    o[k]
    ╱   ℯ
   ╱
   ‾‾‾
  k = 0

because:
    y[k]=0, where k!=j

    partial derivative wrt o_j of Sum_k e^o_j = e^o_j
"""

doj_loss = exp(o[j])/Sum(exp(o[k]), k) - y[j]

diff(doj_loss, o[j])
"""
          q
         ____
         ╲
          ╲
   o[j]    ╲   o[k]
  ℯ    ⋅   ╱  ℯ    ⋅δ
          ╱          j,k
         ╱
         ‾‾‾‾                  o[j]
        k = 1                 ℯ
- ────────────────────── + ───────────
                   2         q
      ⎛  q        ⎞         ___
      ⎜ ___       ⎟         ╲
      ⎜ ╲         ⎟          ╲    o[k]
      ⎜  ╲    o[k]⎟          ╱   ℯ
      ⎜  ╱   ℯ    ⎟         ╱
      ⎜ ╱         ⎟         ‾‾‾
      ⎜ ‾‾‾       ⎟        k = 1
      ⎝k = 1      ⎠

simplifies to (as described above):
-exp(o[j])**2/Sum(o[k], k)**2 + exp(o[j])/Sum(exp(o[k]), k):

      2⋅o[j]          o[j]
     ℯ               ℯ
- ───────────── + ───────────
              2     q
  ⎛  q       ⎞     ___
  ⎜ ___      ⎟     ╲
  ⎜ ╲        ⎟      ╲    o[k]
  ⎜  ╲       ⎟      ╱   ℯ
  ⎜  ╱   o[k]⎟     ╱
  ⎜ ╱        ⎟     ‾‾‾
  ⎜ ‾‾‾      ⎟    k = 1
  ⎝k = 1     ⎠


equivalent to:
exp(o[j])/Sum(o[k], k) * (1 - exp(o[j])/Sum(o[k], k))

or:
softmax*(1-softmax)

THUS:
d**2/d_o[j](loss) = softmax(o)[j]*(1-softmax(o)[j])
"""









# Q 1.2. Compute the variance of the distribution given by  softmax(𝐨)  and show that it matches the second derivative computed above.

from sympy.stats import ContinuousRV, DiscreteRV, P, E
from sympy.stats import variance as V

X = DiscreteRV(x, exp(x)/Sum(exp(x), k))

# E(X)
"""
  ∞
_____
╲
 ╲
  ╲    ⎧x
   ╲   ⎪─  for x = ⌊x⌋ ∧ x < ∞
   ╱   ⎨q
  ╱    ⎪
 ╱     ⎩0       otherwise
╱
‾‾‾‾‾
x = -∞
"""

# V(X)
"""
    ∞
__________
╲
 ╲
  ╲        ⎧              2
   ╲       ⎪⎛      ∞     ⎞
    ╲      ⎪⎜     ____   ⎟
     ╲     ⎪⎜     ╲      ⎟
      ╲    ⎪⎜      ╲     ⎟
       ╲   ⎪⎜       ╲   x⎟
        ╲  ⎪⎜x -    ╱   ─⎟
        ╱  ⎨⎜      ╱    q⎟
       ╱   ⎪⎜     ╱      ⎟
      ╱    ⎪⎜     ‾‾‾‾   ⎟
     ╱     ⎪⎝    x = -∞  ⎠
    ╱      ⎪───────────────  for x = ⌊x⌋ ∧ x < ∞
   ╱       ⎪       q
  ╱        ⎪
 ╱         ⎩       0              otherwise
╱
‾‾‾‾‾‾‾‾‾‾
  x = -∞
"""
# not giving us much to go on

# retry from where we left off in 1.1.
sm = symbols('sm', cls=Function)

sm = exp(x)/Integral(exp(x))

""" attempt #2
Start with second derivative and transform it into the variance

d**2/d_o[j](loss) = softmax(o)[j]*(1-softmax(o)[j])

Expectation of softmax(𝐨)
E[softmax(𝐨)]= Sum(softmax(o), k)/k
"""


"""
2. Assume that we have three classes which occur with equal probability, i.e., the probability vector is  (13,13,13) .

    1.What is the problem if we try to design a binary code for it?

    2.Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode  𝑛  observations jointly?
-----------------------------------------------------------------------------------------------

2.1.
One of the fundamental theorems of information theory states that in order to encode
data drawn randomly from the distribution  𝑃 , we need at least  𝐻[𝑃]  “nats” to encode
it. If you wonder what a “nat” is, it is the equivalent of bit but when using a code
with base  𝑒  rather than one with base 2. Thus, one nat is  1/log(2)≈1.44  bit.

P ~ Uniform([1/3, 1/3, 1/3])

So by the information theory
𝐻[𝑃] = ∑_𝑗 −𝑃(𝑗)log𝑃(𝑗)

We need 1.09 nats to encode the distribution P:

[ins] In [428]: -1/3*log(1/3)*3
Out[428]: 1.09861228866811

Which comes out to 1.58 bits:
[nav] In [431]: -1/3*log(1/3)*3*1.44
Out[431]: 1.58200169568208

Therefore binary (1 bit) is not enough capacity with which to encode the distribution.


2.2.
2-bits are the minimum bits to losslessly encode a single sample from P.
If we wish to encode n samples, then we require 2n bits.
"""

"""
3. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as  RealSoftMax(𝑎,𝑏)=log(exp(𝑎)+exp(𝑏)) .

    1. Prove that  RealSoftMax(𝑎,𝑏)>max(𝑎,𝑏) .

    2. Prove that this holds for  1/𝜆*RealSoftMax(𝜆𝑎,𝜆𝑏) , provided that  𝜆>0 .

    3. Show that for  𝜆→ ∞  we have  1/𝜆*RealSoftMax(𝜆𝑎,𝜆𝑏)→ max(𝑎,𝑏) .

    4. What does the soft-min look like?

    5. Extend this to more than two numbers.
-----------------------------------------------------------------------------------------------

"""

# 3.1. Prove that  RealSoftMax(𝑎,𝑏)>max(𝑎,𝑏) .
a, b = symbols('a b')
StrictGreaterThan(log(exp(a) + exp(b)) , Max(a, b))
"""
       ⎛ a    b⎞
    log⎝ℯ  + ℯ ⎠ > Max(a, b)

    assume (a > b)

    Max(a,b) -> a:

       ⎛ a    b⎞
    log⎝ℯ  + ℯ ⎠ > a

    (exponentiate both sides)

     a    b    a
    ℯ  + ℯ  > ℯ

    True
"""

# 2. Prove that this holds for  1/𝜆*RealSoftMax(𝜆𝑎,𝜆𝑏) , provided that  𝜆>0 .

a, b, 𝜆 = symbols('a b 𝜆')
StrictGreaterThan( 1/𝜆*log(𝜆*exp(a) + 𝜆*exp(b)) , Max(a, b))

"""
Show:
   ⎛   a      b⎞
log⎝𝜆⋅ℯ  + 𝜆⋅ℯ ⎠
──────────────── > Max(a, b)
       𝜆

Assume:
    a > b
    𝜆 > 0

(Max(a,b) -> a, b/c a > b)

   ⎛   a      b⎞
log⎝𝜆⋅ℯ  + 𝜆⋅ℯ ⎠
──────────────── > a
       𝜆

   ⎛   a      b⎞
log⎝𝜆⋅ℯ  + 𝜆⋅ℯ ⎠ > 𝜆a

(exp both sides)

   a      b    𝜆a
𝜆⋅ℯ  + 𝜆⋅ℯ  > ℯ

 a    b        𝜆a
ℯ  + ℯ  > 1/𝜆⋅ℯ

"""

3. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as  RealSoftMax(𝑎,𝑏)=log(exp(𝑎)+exp(𝑏)) .

    2. Prove that this holds for  1/𝜆*RealSoftMax(𝜆𝑎,𝜆𝑏) , provided that  𝜆>0 .

    3. Show that for  𝜆→ ∞  we have  1/𝜆*RealSoftMax(𝜆𝑎,𝜆𝑏)→ max(𝑎,𝑏) .
