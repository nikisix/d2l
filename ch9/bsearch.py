from itertools import chain
import numpy as np


class BinarySearch:
    """ __call__ Params:
    f_1 - previous function call value
    s - list of slices (one per variable to optimize)
    d - dimension to optmize over
    Returns: min function value over the slices
    """
    def __init__(self, fn, opt_params, min_calls=5, debug=False):
        """
        fn - function to optimize
        opt_params - params to optimize over
        min_calls - min number of calls to make before killing a search space
        """
        self.params = dict()
        self.calls = 0  # num_calls, break at max
        self.fn = fn
        epsilon = []
        for p in opt_params:
            if p.step:
                epsilon.append(p.step)
            else:
                epsilon.append(1)
        self.epsilon = epsilon
        self.min_calls = min_calls
        self.debug = debug

    def __call__(self, f_1, s, d):
        """
        f_1 (float) - function value from previous run
        s (slice) - search space
        d (int) - dimension to search
        """
        self.calls += 1

        x=list()
        for i in range(len(s)):
            if self.epsilon[i] >= 1:
                x.append(int((s[i].stop + s[i].start)/2))
            else:
                x.append((s[i].stop + s[i].start)/2)
        fx = np.abs(self.fn(*x))  # cache money
        self.params[fx] = list(x)
        if self.debug: print(fx, f_1, '\t', x)
        if (f_1 < fx) and (self.min_calls < self.calls): return
        if any([s[i].stop - s[i].start <= self.epsilon[i] for i in range(len(s))]): 
            return fx

        s_left, s_right = s.copy(), s.copy()
        # "right-bound" of the split between left and right ranges
        r_bound = [x[d]+1, s[d].stop ][bool(x[d]+1 > s[d].stop)]
        l_bound = [x[d]-1, s[d].start][bool(x[d]-1 < s[d].start)]
        s_right[d] = slice(r_bound, s[d].stop, self.epsilon[d])
        s_left[d]  = slice(s[d].start, l_bound, self.epsilon[d])

        if 1 < len(s):
            return min(filter(lambda o: o!=None,
                chain(
                    [fx],
                    [self.__call__(fx, s_left,  i) for i in range(len(s)) if i!=d],
                    [self.__call__(fx, s_right, i) for i in range(len(s)) if i!=d],
                )
            ))
        else:
            return min(filter(lambda o: o!=None,
                chain(
                    [fx],
                    [self.__call__(fx, s_left, 0)],
                    [self.__call__(fx, s_right, 0)],
                )
            ))
