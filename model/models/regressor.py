import torch
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear


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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
