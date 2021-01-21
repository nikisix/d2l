from sympy import *
from sympy.abc import x

init_printing(use_unicode=True)

""" Q1.
let's look at the function:

            ⎡ 3   1⎤
    f(x) =  ⎢x  - ─⎥
            ⎣     x⎦

[nav] In [11]: plot(x**3 - 1/x, backend='text')
   1000 |                                                      /
        |                                                     /
        |                                                    /
        |                                                   /
        |                                                  /
        |                                                ..
        |                                               /
        |                                             ..
        |                                          ...
        |                                     .....
      0 |------------------.........-.........------------------
        |             .....
        |          ...
        |        ..
        |       /
        |     ..
        |    /
        |   /
        |  /
        | /
  -1000 |_______________________________________________________
         -10                        0                          10

[ins] In [8]: plot(diff(x**3 - 1/x, x), (x, .75, 1.25), backend='text')
    5.3 |                                                     ..
        |                                                    /
        |                                                  ..
        |                                                 /
        |                                               ..
        |                                             ..
        |                                            /
        |                                          ..
        |                                        ..
        |                                      ..
    4.4 |------------------------------------..-----------------
        |                                  ..
        |                                ..
        |                              ..
        |                           ...
        |                         ..
        |                      ...
        |                   ...
        |               ....
        |           ....
    3.5 |_______________________________________________________
         0.75                       1                          1.25
Out[8]: <sympy.plotting.plot.Plot at 0x1182707f0>

[ins] In [9]: plot(diff(x**3 - 1/x, x), (x, -1-.25, -1+.25), backend='text')
    5.3 |..
        |  \
        |   ..
        |     \
        |      ..
        |        ..
        |          \
        |           ..
        |             ..
        |               ..
    4.4 |-----------------..------------------------------------
        |                   ..
        |                     ..
        |                       ..
        |                         ...
        |                            ..
        |                              ...
        |                                 ...
        |                                    ....
        |                                        ....
    3.5 |_______________________________________________________
         -1.25                      -1                         -0.75
Out[9]: <sympy.plotting.plot.Plot at 0x1183080d0>

[nav] In [10]: diff(x**3 - 1/x, x)
Out[10]: 3*x**2 + x**(-2)
"""


"""Q2. Find the gradient of :
        ⎡   2    5⋅y⎤
        ⎣3⋅x  + ℯ   ⎦
Ans:
        ⎡        5⋅y⎤
        ⎣6⋅x, 5⋅ℯ   ⎦
"""


"""Q3. What is the gradient of the function 𝑓(𝐱)=‖𝐱‖2
             ________________
            ╱   2     2
l₂Norm := ╲╱  x₁  + x₂  + ...

Ans:

                ⎡      x₁              x₂            ⎤
                ⎢──────────────, ──────────────, ... ⎥
grad(l₂Norm) =  ⎢   ___________     ___________      ⎥
                ⎢  ╱   2     2     ╱   2     2       ⎥
                ⎣╲╱  x₁  + x₂    ╲╱  x₁  + x₂        ⎦
"""


"""Q4. Can you write out the chain rule for the case where
𝑢=𝑓(𝑥,𝑦,𝑧)  and  𝑥=𝑥(𝑎,𝑏),  𝑦=𝑦(𝑎,𝑏), and 𝑧=𝑧(𝑎,𝑏)?

du/dx*dx/da + du/dx/dx/db + du/dy*dy/da

"""
u, x, y, z = symbols('u, x, y, z', cls=Function)
a, b = symbols('a,b')
y=y(a,b)
z=z(a,b)
z=z(a,b)
u = u(x,y,z)


"""
[ins] In [103]: diff(x,a)
Out[103]:
∂
──(x(a, b))
∂a
"""

"""
[nav] In [102]: diff(u, a)
Out[102]:

      ∂                                   ∂
   ────────(u(x(a, b), y(a, b), z(a, b)))⋅──(x(a, b))
   ∂x(a, b)                               ∂a

      ∂                                   ∂
 + ────────(u(x(a, b), y(a, b), z(a, b)))⋅──(y(a, b))
   ∂y(a, b)                               ∂a

      ∂                                   ∂
 + ────────(u(x(a, b), y(a, b), z(a, b)))⋅──(z(a, b))
   ∂z(a, b)                               ∂a
"""
