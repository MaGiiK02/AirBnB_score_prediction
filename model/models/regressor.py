import torch
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.nn import functional as F




import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback

class _MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)


class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.autoencoder = _MLPRegressor(input_dim, hidden_dim)
    
    
    def forward(self, x):
        return self.autoencoder(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('test_loss', loss, on_epoch=True)
        return loss

# Define the PyTorch Lightning Trainer and Callback
class LossAccCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        train_loss = trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss)

    def on_test_epoch_end(self, trainer, pl_module):
        test_loss = trainer.callback_metrics['test_loss'].item()
        self.test_losses.append(test_loss)


