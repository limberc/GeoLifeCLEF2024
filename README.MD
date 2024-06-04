# GeoLifeCLEF2024

The minimal code to reproduce the [GeoLifeCLEF 2024](https://www.kaggle.com/competitions/geolifeclef-2024/) challenge.

```python
import torchvision.models as models
from pytorch_lightning import Trainer

from datamodule import GeoLifeCLEF2024DataModule
from model import GeoLifeCLEF2024

datamodule = GeoLifeCLEF2024DataModule(batch_size=256)
datamodule.setup()
features = datamodule.features
# We setup the `sentinel_model` to be a pre-trained model from torchvision.
sentinel_model = models.convnext_large('IMAGENET1K_V1')
model = GeoLifeCLEF2024(features, sentinel_model, lr=8e-4)
trainer = Trainer(max_epochs=15, precision=16)
trainer.fit(model, datamodule)
```