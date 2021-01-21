"""
derive linear regression analytical solution

https://stats.stackexchange.com/questions/336860/derivation-of-the-closed-form-solution-to-minimizing-the-least-squares-cost-func

matrix cookbook: http://www2.imm.dtu.dk/pubdb/edoc/imm3274.pdf
"""

# Q1
"""Assume that we have some data  𝑥1,…,𝑥𝑛∈ℝ . Our goal is to find a constant  𝑏  such that  ∑𝑖(𝑥𝑖−𝑏)2  is minimized.

1. Find a analytic solution for the optimal value of  𝑏 .

2. How does this problem and its solution relate to the normal distribution?
----------------------------------------------------------------------------
----------------------------------------------------------------------------
1. Find a analytic solution for the optimal value of  𝑏 .

    Let db indicate the derivative wrt b

    0 = db(∑(x-b)^2)

    0 = ∑( db(x-b)^2 )

    0 = ∑( 2*(x-b)*db(x-b) )

    0 = ∑( 2*(x-b)*(0 - db(b)) )

    0 = ∑( 2*(x-b)*(-1) )

    0 = ∑( -2*(x-b) )

    0 = -2*∑(x-b)

    0 = ∑(x-b)

    0 = ∑(x) - ∑(b)

    Assume x has n values

    0 = ∑(x) - nb

    nb = ∑(x)

    b = ∑(x)/n


2. This is the mean of x.
    The normal contains the term (x - mu)^2.
    Mu is the mean of the Normal, just as the optimal b is the mean of x.
"""

# Q2
"""
Derive the analytic solution to the optimization problem for linear regression with squared error. To keep things simple, you can omit the bias  𝑏  from the problem (we can do this in principled fashion by adding one column to  𝐗  consisting of all ones).

1. Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, and all the target values as a single vector).

2. Compute the gradient of the loss with respect to  𝑤 .

3. Find the analytic solution by setting the gradient equal to zero and solving the matrix equation.

4. When might this be better than using stochastic gradient descent? When might this method break?
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
1. Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, and all the target values as a single vector).
2. Compute the gradient of the loss with respect to  𝑤 .
3. Find the analytic solution by setting the gradient equal to zero and solving the matrix equation.

    NOTATION:
    w, y are vectors, X is a matrix
    wt: "w transpose"
    dw: "derivative wrt w"
    a space is inserted between terms for clarity

    0 = min(loss(w))

    0 = min((y-wX)^2)

    simplify, then add back in the 0 = dw()

    (y - wX)^2

    (y - wX)t*(y - wX)

    (yt - wt*Xt)*(y - wX)

    yt y - wt Xt y - yt w X + wt Xt w X

    ( wt Xt y is scalar.. so (wt Xt y) = (wt Xt y)t )

    simplified expression:
    yt y - 2 yt w X + Xt X w^2

    Now, take derivative and set = to 0
    0 = dw(yt y - 2 yt w X + Xt X w^2)

    0 = 0 - 2 yt X + 2 Xt X w

    solve for w

    2 yt X = 2 Xt X w

    Xt X w = yt X

    w = yt X (Xt X)^-1

    from text:
    𝐰∗ = (𝐗⊤𝐗)−1𝐗⊤𝐲.

    ? is (yt X) == (Xt y) ?


4. When might this be better than using stochastic gradient descent? When might this method break?

    Better when the cube of the dataset ( (𝐗⊤𝐗)−1𝐗⊤ ) fits into memory. Breaks when it does not.
"""


# Q3
"""Assume that the noise model governing the additive noise  𝜖  is the exponential distribution. That is,  𝑝(𝜖)=1/2 exp(−|𝜖|) .

1. Write out the negative log-likelihood of the data under the model  −log𝑃(𝐲∣𝐗) .

2. Can you find a closed form solution?

3. Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?
----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- --------------
----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- ----------------------- --------------
1. Write out the negative log-likelihood of the data under the model  −log𝑃(𝐲∣𝐗) .

    𝑝(𝜖)=1/2 exp(−|𝜖|)

    1.1.
    p(y|X) = 1/2 exp( -abs(y - wt x - b ) )

    1.2
    log p(y|X) = log(1/2) - abs(y - wt x - b)

    -log p(y|X) = -log(1/2) + abs(y - wt x - b)


2. Can you find a closed form solution?

    Strategy: Derive wrt w and set to 0

    " 0 = ... "
    dw (-log p(y|X)) = 0

    dw (-log p(y|X)) = dw (-log(1/2) + abs(y - wt x - b))

    dw (-log(1/2) + abs(y - wt x - b))

    0 + dw abs(y - wt x - b)

    abs(dw y - dw wt x - dw b)

    abs(0 - x - 0)

    abs(-x)

    "set = 0"

    0 = abs(-x)

    0 != abs(-x)

    No solution exists.

    Solution doesn't exist b/c stable point isn't well formed (comes to a point):

    [ins] In [19]: plot(.5*exp(-abs(x)), backend='text')
    0.5 |                           .
        |
        |
        |
        |
        |
        |                          . .
        |
        |
        |
   0.25 |-------------------------.---.-------------------------
        |
        |
        |                        .     .
        |
        |                       .       .
        |
        |                      .         .
        |                     /           \
        |                  ...             ...
      0 |_______________________________________________________
         -10                        0                          10


3. Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?

    Alternate back and fourth around stationary point
    Decrease step size ("learning rate")
    Discretize the range
"""




































