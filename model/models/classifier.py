import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

# An easily scalable MLP model using ReLu activation
class MLPClassifier(nn.Module):
    def __init__(self, layers=[4, 4]):
        super().__init__()
        self.layers=[]
        for i in range(len(layers)-1):
            layer = torch.Sequential(
                Linear(layers[1], layers[i+1], bias=False, dtype=torch.float)
                ReLU()
            )
            self.layers.append(layer)

    def forward(self, x):

        for l in self.layers:
            x = l(x)
        
        return x
