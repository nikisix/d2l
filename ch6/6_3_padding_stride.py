# Padding and stride formula
# ⌊(n_h−k_h+p_h+s_h)/s_h⌋×⌊(n_w−k_w+p_w+s_w)/s_w⌋
import math
import numpy as np

np.set_printoptions(precision=2)

X = np.random.uniform(size=(8,8))
"""
array([[0.83, 0.69, 0.71, 0.23, 0.53, 0.99, 0.41, 0.48],
       [0.76, 0.5 , 0.19, 0.83, 0.49, 0.35, 0.45, 0.05],
       [0.54, 0.33, 0.62, 0.84, 0.73, 0.81, 0.64, 0.56],
       [0.66, 0.38, 0.42, 0.14, 0.04, 0.14, 0.94, 0.95],
       [1.  , 0.51, 0.96, 0.6 , 0.06, 0.8 , 0.94, 0.95],
       [0.57, 0.6 , 0.68, 0.12, 0.09, 0.26, 0.79, 0.98],
       [0.68, 0.98, 0.61, 0.76, 0.07, 0.79, 0.56, 0.87],
       [0.24, 0.72, 0.66, 0.53, 0.28, 0.56, 0.76, 0.44]])
"""
# conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
n_h = 8; n_w = 8
k_h = 3; k_w = 5
p_h, p_w = 0, 1
s_h, s_w = 3, 4
print(math.floor((n_h-k_h+p_h+s_h)/s_h))
print(math.floor((n_w-k_w+p_w+s_w)/s_w))
"""
    2
    2
"""
