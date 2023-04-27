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
    def __init__(self, numerical_in=60, embeddings_in=1536):
        super().__init__()
        
        self.numerical_in = numerical_in
        self.embeddings_in = embeddings_in

        self.numerical = nn.Sequential(
            nn.Linear(numerical_in, int(numerical_in/2)),
            nn.ReLU(),
            nn.Linear(int(numerical_in/2), int(numerical_in/4)),
            nn.ReLU(),
        )

        self.embeddings = nn.Sequential(
            nn.Linear(embeddings_in, int(embeddings_in/2)), 
            nn.ReLU(),
            nn.Linear(int(embeddings_in/2), int(embeddings_in/4)),
            nn.ReLU(),
            nn.Linear(int(embeddings_in/4), int(embeddings_in/8)),
            nn.ReLU(),
        )

        self.inner_concat_size = int(numerical_in/4) + int(embeddings_in/8)
        self.regression = nn.Sequential(
            nn.Linear(self.inner_concat_size, int(self.inner_concat_size/2)), 
            nn.ReLU(),
            nn.Linear(int(self.inner_concat_size/2), int(self.inner_concat_size/4)), 
            nn.ReLU(),
            nn.Linear(int(self.inner_concat_size/4), 1),
        )

    def forward(self, x):
        x_numerical = x[:, :self.numerical_in]
        x_embeddings = x[:, self.numerical_in:]

        x_numerical =  self.numerical(x_numerical)
        x_embeddings =  self.embeddings(x_embeddings)

        x_concat = torch.cat((x_numerical, x_embeddings), 1)
        return self.regression(x_concat)



class MLPRegressor(pl.LightningModule):
    def __init__(self, numerical_in=60, embeddings_in=1536):
        super().__init__()
        self.model = _MLPRegressor(numerical_in, embeddings_in)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self, lr=1e-6):
        optimizer = optim.AdamW(self.parameters(), lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y.unsqueeze(1))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

# Define the PyTorch Lightning Trainer and Callback
class LossAccCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_losses.append(train_loss)
        self.logger.experiment.add_scalar('Loss/Train', train_loss, self.current_epoch)
        
    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_losses.append(test_loss)
        self.logger.experiment.add_scalar('Loss/Test', test_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        test_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_losses.append(test_loss)
        self.logger.experiment.add_scalar('Loss/Validation', test_loss, self.current_epoch)
