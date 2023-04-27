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
    def __init__(self, input_dim, hidden_dim, model=None):
        super().__init__()
        self.autoencoder = _MLPRegressor(input_dim, hidden_dim) if model == None else model
    
    
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

        

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(train_loss)
        self.logger.experiment.add_scalar('Loss/Train', train_loss, self.current_epoch)
        
    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_losses.append(test_loss)
        self.logger.experiment.add_scalar('Loss/Test', test_loss, self.current_epoch)
