import torchvision.models as models
from pytorch_lightning import Trainer

from datamodule import GeoLifeCLEF2024DataModule
from model import GeoLifeCLEF2024

if __name__ == '__main__':
    # Setup datamodule
    datamodule = GeoLifeCLEF2024DataModule(batch_size=256)
    datamodule.setup()
    features = datamodule.features
    sentinel_model = models.convnext_large('IMAGENET1K_V1')
    model = GeoLifeCLEF2024(features, sentinel_model, lr=8e-4)
    trainer = Trainer(max_epochs=15, precision=16)
