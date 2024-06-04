import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule


class MultimodalEnsemble(nn.Module):
    def __init__(self, num_classes, features, sentinel_model):
        super(MultimodalEnsemble, self).__init__()
        self.tab_norm = nn.LayerNorm([len(features)])
        self.tab_model = nn.Sequential(nn.Linear(len(features), 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 32), )

        self.landsat_norm = nn.LayerNorm([6, 4, 21])
        self.landsat_model = models.resnet18(weights=None)
        # Modify the first convolutional layer to accept 6 channels instead of 3
        self.landsat_model.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.landsat_model.maxpool = nn.Identity()

        self.bioclim_norm = nn.LayerNorm([4, 19, 12])
        self.bioclim_model = models.resnet18(weights=None)
        # Modify the first convolutional layer to accept 4 channels instead of 3
        self.bioclim_model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bioclim_model.maxpool = nn.Identity()

        self.sentinel_model = sentinel_model
        # Modify the first layer to accept 4 channels instead of 3
        self.sentinel_model.features[0][0] = nn.Conv2d(4, 192, kernel_size=(4, 4), stride=(4, 4))
        self.sentinel_model.head = nn.Identity()

        self.ln0 = nn.LayerNorm(32)
        self.ln1 = nn.LayerNorm(1000)
        self.ln2 = nn.LayerNorm(1000)
        self.ln3 = nn.LayerNorm(1000)

        self.fc1 = nn.Linear(3000 + 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, t, x, y, z):
        t = self.tab_norm(t)
        t = self.tab_model(t)
        t = self.ln0(t)
        t = self.dropout(t)

        x = self.landsat_norm(x)
        x = self.landsat_model(x)
        x = self.ln1(x)
        x = self.dropout(x)

        y = self.bioclim_norm(y)
        y = self.bioclim_model(y)
        y = self.ln2(y)
        y = self.dropout(y)

        z = self.sentinel_model(z)
        z = self.ln3(z)
        z = self.dropout(z)

        txyz = torch.cat((t, x, y, z), dim=1)

        txyz = self.fc1(txyz).relu()
        txyz = self.dropout(txyz)

        out = self.fc2(txyz)
        return out


class GeoLifeCLEF2024(LightningModule):
    def __init__(self, features, sentinel_model, lr=8e-4):
        super(GeoLifeCLEF2024, self).__init__()
        self.model = MultimodalEnsemble(num_classes=11255, features=features,
                                        sentinel_model=sentinel_model)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        tab, landsat_sample, bioclim_sample, sentinel_sample, targets, _ = batch
        # Mixup
        if np.random.rand() < 0.4:
            lam = torch.tensor(np.random.beta(0.4, 0.4))
            rand_index = torch.randperm(tab.size()[0])
            mixed_tab = lam * tab + (1 - lam) * tab[rand_index]
            mixed_landsat_sample = lam * landsat_sample + (1 - lam) * landsat_sample[rand_index]
            mixed_bioclim_sample = lam * bioclim_sample + (1 - lam) * bioclim_sample[rand_index]
            mixed_sentinel_sample = lam * sentinel_sample + (1 - lam) * sentinel_sample[rand_index]
            targets_a, targets_b = targets, targets[rand_index]
            mixed_targets = lam * targets_a + (1 - lam) * targets_b
            outputs = self(mixed_tab, mixed_landsat_sample, mixed_bioclim_sample, mixed_sentinel_sample)
            loss = self.criterion(outputs, mixed_targets)
        else:
            outputs = self(tab, landsat_sample, bioclim_sample, sentinel_sample)
            loss = self.criterion(outputs, targets)
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tab, landsat_sample, bioclim_sample, sentinel_sample, targets, _ = batch
        outputs = self(tab, landsat_sample, bioclim_sample, sentinel_sample)
        loss = self.criterion(outputs, targets)
        self.log('val/loss', loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def forward(self, tab, landsat_sample, bioclim_sample, sentinel_sample, targets):
        return self.model(tab, landsat_sample, bioclim_sample, sentinel_sample)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
