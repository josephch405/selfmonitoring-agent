import torch
from torch import nn
from torch import functional as F

final_output_dims = [10, 3]

class LinearBinaryModel(nn.Module):
    def __init__(self, *sizes, activation=nn.ReLU):
        super(LinearBinaryModel, self).__init__()
        assert len(sizes) >= 1, "Need at least one 'sizes' specified"
        sizes = list(sizes) + [1]
        self.layers = [nn.Linear(sizes[0], sizes[1])]
        i = 1
        for i in range(1, len(sizes) - 1):
            self.layers.append(activation())
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            i += 1
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        # x is [-1, 2048]
        y = x
        y = self.layers(y).sigmoid()
        return y

class LinearModel(nn.Module):
    def __init__(self, *sizes, activation=nn.ReLU):
        super(LinearModel, self).__init__()
        assert len(sizes) >= 1, "Need at least one 'sizes' specified"
        sizes = list(sizes) + [30]
        self.layers = [nn.Linear(sizes[0], sizes[1])]
        i = 1
        for i in range(1, len(sizes) - 1):
            self.layers.append(activation())
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
            i += 1
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        # x is [-1, 2048]
        y = x
        y = self.layers(y)
        y = y.view(-1, 10, 3)

        y[:, :, :2] = y[:, :, :2].tanh()
        y[:, :, 2] = y[:, :, 2].sigmoid() * 20

        mask = y[:, :, 2] > 0.5
        y[:, :, 0] *= mask
        y[:, :, 1] *= mask

        return y

        # returns [-1, 10, 3]