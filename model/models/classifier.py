import torch
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

# An easily scalable MLP model using ReLu activation
class _MLPClassifier(nn.Module):
    def __init__(self, in_size, classes):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_size, int(in_size/2)), 
            nn.ReLU(),
            nn.Linear(int(in_size/2), int(in_size/4)),
            nn.ReLU(),
            nn.Linear(int(in_size/4), int(classes)),
            nn.ReLU(),
        )

    def forward(self, x):
        
        return self.seq(x)


class MLPClassifier(pl.LightningModule):
    def __init__(self, in_size, classes):
        super().__init__()
        self.model = _MLPClassifier(in_size, classes)
    
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return torch.argmax(self.model(x), dim=0)

    def configure_optimizers(self, lr=1e-6):
        optimizer = optim.AdamW(self.parameters(), lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat.softmax(dim=1), y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat.softmax(dim=1), y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.functional.cross_entropy(y_hat.softmax(dim=1), y)
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