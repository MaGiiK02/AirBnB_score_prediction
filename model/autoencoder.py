import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

class _Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def decode(self, x):
        return self.decoder(x)
    
    def encoder(self, x):
        return self.encoder(x)

# How to train example
# train_loader = utils.data.DataLoader(dataset) # Any pytorch dataset
# trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(model=autoencoder, train_dataloaders=train_loader)
class Autoencoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.autoencoder = _Autoencoder(
            input_size=input_size, hidden_size=hidden_size
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self.autoencoder(x)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def encode(self, x):
        self.autoencoder.encoder(x)

    def decode(self):
        self.decode.encoder(x)