from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
# x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
