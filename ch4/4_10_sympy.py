from sympy import *
from sympy.abc import n, i
import sympy

y = sympy.IndexedBase("y")
y_hat = sympy.IndexedBase("y_hat")
i = Idx("i", (1, n))
sympy.init_printing(use_unicode=True)
sqrt(1/n * Sum((log(y) - log(y_hat))**2, i))
