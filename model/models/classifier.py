import torch
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

# An easily scalable MLP model using ReLu activation
class _MLPClassifier(nn.Module):
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


class MLPClassifier(pl.LightningModule):
    def __init__(self, layers):
        super().__init__()
        self.autoencoder = _MLPClassifier(layers=layers)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self.autoencoder(x)
        loss = nn.functional.cross_entropy(x_hat.softmax(dim=1), y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer