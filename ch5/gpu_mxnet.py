from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

print(npx.gpu(0))

def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)

print(gpu_device)


################################################################################
# NNs on gpus
################################################################################

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=npx.gpu())

X = np.ones((2, 3), ctx=npx.gpu())

print(net(X))

"""
Exercises

1. Try a larger computation task, such as the multiplication of large matrices, and see the difference in speed between the
CPU and GPU. What about a task with a small amount of calculations?

2. How should we read and write model parameters on the GPU?

3. Measure the time it takes to compute 1000 matrix-matrix multiplications of 100×100 matrices and log the Frobenius norm
of the output matrix one result at a time vs. keeping a log on the GPU and transferring only the final result.

4. Measure how much time it takes to perform two matrix-matrix multiplications on two GPUs at the same time vs. in sequence
on one GPU. Hint: you should see almost linear scaling.
"""

# 3. Measure the time it takes to compute 1000 matrix-matrix multiplications of 100×100 matrices and log the Frobenius norm
# of the output matrix one result at a time vs. keeping a log on the GPU and transferring only the final result.
from d2l import torch as d2l
X = np.ones((10000, 10000), ctx=npx.gpu())
# X = np.ones((10000, 10000), ctx=npx.cpu())

t0 = d2l.time.perf_counter()
for i in range(1000):
    # np.sqrt(np.sum((X*X)**2))
    # np.linalg.norm(X*X)
    np.linalg.norm(np.matmul(X,X))
t1 = d2l.time.perf_counter()
print(f'clocked: {t1-t0}')

# for 100x100 matricies:
# gpu clocked: 0.08214004000183195
# cpu clocked: 0.07641711199903511

# for 10000x10000 matricies:
# gpu clocked: 0.07812751299934462
# cpu clocked: 0.1442705420013226
