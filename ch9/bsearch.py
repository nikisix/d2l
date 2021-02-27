from itertools import chain

#TODO delete
import numpy as np
import random
from itertools import chain


class BinarySearch:
    """ __call__ Params:
    f_1 - previous function call value
    s - list of slices (one per variable to optimize)
    d - dimension to optmize over
    Returns: min function value over the slices
    """
    def __init__(self, fn, epsilon=1, debug=False):
        """
        fn - function to optimize
        epsilon - search stopping granularity
        """
        self.params = dict()
        self.calls = 0  # num_calls, break at max
        self.fn = fn
        self.epsilon = epsilon
        self.debug = debug

    def __call__(self, f_1, s, d):
        return self.bsearch(f_1, s, d)

    def bsearch(self, f_1, s, d):
        self.calls += 1

        x=list()
        for i in range(len(s)):
            x.append(int((s[i].stop + s[i].start)/2))
        fx = self.fn(lr=x[0])  # cache money
        # if self.debug: print(x, fx)
        self.params[fx] = list(x)
        if self.debug: print(fx, f_1, '\t', x)
        if f_1 < fx: return
        if any([s[i].stop - s[i].start <= self.epsilon for i in range(len(s))]): 
            return fx

        s_left, s_right = s.copy(), s.copy()
        r_bound = [x[d]+1, s[d].stop][bool(x[d]+1>s[d].stop)]
        l_bound = [x[d]-1, s[d].start][bool(x[d]-1 < s[d].start)]
        s_right[d] = slice(r_bound, s[d].stop, 1)
        s_left[d] = slice(s[d].start, l_bound, 1)

        return min(filter(lambda o: o!=None,
            chain( 
                [fx],
                [self.bsearch(fx, s_left, i) for i in range(len(s)) if i!=d],
                [self.bsearch(fx, s_right, i) for i in range(len(s)) if i!=d],
            )
        ))

params = dict()
num_calls = 0
def bsearch1(f_1, s):
    """Binary search over a single variable
    f_1: previous function value
    s: current search range"""
    global params
    global num_calls
    num_calls += 1
    x = (s.stop + s.start)/2
    fx = f(x)
    if fx < f_1: return
    if s.stop - s.start <= epsilon:
        params[x] = fx
        return fx
    return max(filter(lambda o: o!=None, [
        fx,
        bsearch1(fx, slice(x+1, s.stop)), # right
        bsearch1(fx, slice(s.start, x-1)),# left
    ]))

f = lambda x: -(x-4)**2 + 100
s = slice(-10, 10, 1) # search range
x_init = random.randrange(s.start, s.stop)
print(bsearch1(f(x_init), s), 'searched maximum')
print(, 'function calls')
x_hat = list(params.keys())[np.argmax(np.array(list(params.values())))]
print(x_hat, 'maximizes f over the range')
