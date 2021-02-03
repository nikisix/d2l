# f = lambda x, y: x+y

"""
[ins] In [11]: sympy.plot(-x**2 + 100, backend='text')
    100 |                       .........
        |                    ...         ...
        |                  ..               ..
        |                ..                   ..
        |               /                       \
        |             ..                         ..
        |            /                             \
        |           /                               \
        |          /                                 \
        |         /                                   \
     50 |--------/-------------------------------------\--------
        |       /                                       \
        |      /                                         \
        |     .                                           .
        |
        |    .                                             .
        |   /                                               \
        |  .                                                 .
        |
        | .                                                   .
      0 |_______________________________________________________
         -10                        0                          10
"""

import numpy as np
import random
from itertools import chain

# f = lambda x: -(x-4)**2 + 100
# epsilon = 1
# num_calls=0
# params = dict()
# """Binary search over a single variable
# f_1: previous function value
# s: current search range"""
# def bsearch(f_1, s):
    # global params
    # global num_calls
    # num_calls += 1
    # x = (s.stop + s.start)/2
    # fx = f(x)
    # if fx < f_1: return
    # if s.stop - s.start <= epsilon:
        # params[x] = fx
        # return fx
    # return max(filter(lambda o: o!=None, [
        # fx,
        # bsearch(fx, slice(x+1, s.stop)), # right
        # bsearch(fx, slice(s.start, x-1)),# left
    # ]))

# s = slice(-10, 10, 1) # search range
# x_init = random.randrange(s.start, s.stop)
# print(bsearch(f(x_init), s), 'searched maximum')
# print(num_calls, 'function calls')
# x_hat = list(params.keys())[np.argmax(np.array(list(params.values())))]
# print(x_hat, 'maximizes f over the range')


# f2 = lambda x, y: -(x+y)**2 + 100
# def bsearch2(f_1, s1, s2, i):
    # """Binary search over two variables"""
    # global num_calls
    # num_calls += 1
    # global params
    # i = [0, 1][(i+1)%2]  # 0, 1, 0, 1 ...etc
    # x1 = (s1.stop + s1.start)/2
    # x2 = (s2.stop + s2.start)/2
    # fx1x2 = f2(x1, x2)  # cache money
    # params[(x1, x2)] = fx1x2
    # if s1.stop - s1.start <= epsilon: return fx1x2
    # if s2.stop - s2.start <= epsilon: return fx1x2
    # if i: # search in x1
        # rhs = bsearch2(fx1x2, slice(x1+1, s1.stop), s2, i)
        # lhs = bsearch2(fx1x2, slice(s1.start, x1-1), s2, i)
        # return np.max([fx1x2, rhs, lhs])
    # else: # search in x2
        # rhs = bsearch2(fx1x2, s1, slice(x2+1, s2.stop), i)
        # lhs = bsearch2(fx1x2, s1, slice(s2.start, x2-1), i)
        # return np.max([fx1x2, rhs, lhs])

# s = slice(-10, 10)
# x1_init = random.randrange(s.start, s.stop)
# x2_init = random.randrange(s.start, s.stop)
# print(x1_init, x2_init)
# print(bsearch2(f2(x1_init, x2_init), s, s, 0))
# x_hat = list(params.keys())[np.argmax(np.array(list(params.values())))]
# print(x_hat, 'maximizes f over the range')
# print(num_calls, 'function calls')


# General Case of n-variables to search over
# """Binary search over two variables
# f_1: parent's function value
# s: list of ranges to search over
# d: dimension we're optimizing"""
fn = lambda x: -(x.sum()-3)**2 + 100
epsilon = 1
calls=0
params=dict()
def bsearch(f_1, s, d):
    global calls
    global params
    calls += 1

    x=list()
    for i in range(len(s)):
        x.append((s[i].stop + s[i].start)/2)
    x = np.array(x)
    fx = fn(x)  # cache money
    print(fx,'\t', x, '\t', s)
    params[fx] = list(x)
    if fx < f_1: return
    if any([s[i].stop - s[i].start <= epsilon for i in range(len(s))]): 
        return fx

    s_left, s_right = s.copy(), s.copy()
    r_bound = [x[d]+1, s[d].stop][bool(x[d]+1>s[d].stop)]
    l_bound = [x[d]-1, s[d].start][bool(x[d]-1 < s[d].start)]
    s_right[d] = slice(r_bound, s[d].stop, 1)
    s_left[d] = slice(s[d].start, l_bound, 1)

    # print(f'{s}->\t{s_left}\t{s_right}')
    return max(filter(lambda o: o!=None,
        chain( 
            [fx],
            [bsearch(fx, s_left, i) for i in range(len(s)) if i!=d],
            [bsearch(fx, s_right, i) for i in range(len(s)) if i!=d],
        )
    ))

# Searches one dimension over only
# rhs = bsearch2(fx, s_left, (d+1)%len(s))
# lhs = bsearch2(fx, s_right,(d+1)%len(s))

low, high = -10, 10
dim = 2
s = slice(low, high, 1)
x = np.random.uniform(s.start, s.stop, size=dim)
print(x, 'X-init')
print(bsearch(fn(x), [s]*dim, 0))
print(params[max(params)], 'maximizes f over the range')
print(calls, 'function calls')
