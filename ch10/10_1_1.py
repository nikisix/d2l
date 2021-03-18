from heatmap import show_heatmaps
import matplotlib as mpl
import torch

# random tensor
rt = torch.rand((10, 10))
smt = torch.nn.Softmax(dim=0)(rt).reshape(1, 1, 10, 10)
# smt = torch.eye(10).reshape((1, 1, 10, 10))  # identity example

show_heatmaps(smt, xlabel='Tensor', ylabel='something')
