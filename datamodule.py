import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

from data import TestDataset, TrainDataset


def set_seed(seed):
    # Set seed for Python's built-in random number generator
    torch.manual_seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GeoLifeCLEF2024DataModule(LightningDataModule):
    def __init__(self, root_dir='./geolifeclef-2024/', batch_size=256, train_ratio=0.9, seed=666):
        super().__init__()
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.seed = seed
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        set_seed(self.seed)
        env_raster_path = 'EnvironmentalRasters/EnvironmentalRasters'
        train_landcover = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/LandCover/GLC24-PA-train-landcover.csv")
        train_solidgrids = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/SoilGrids/GLC24-PA-train-soilgrids.csv")
        train_humanfootprint = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Human Footprint/GLC24-PA-train-human_footprint.csv")
        train_elevation = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Elevation/GLC24-PA-train-elevation.csv")
        train_climate = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Climate/Average 1981-2010/GLC24-PA-train-bioclimatic.csv")
        # Merge all the environmental rasters data.
        train_tab = train_climate.merge(
            train_elevation, on="surveyId").merge(
            train_humanfootprint, on="surveyId").merge(
            train_solidgrids, on="surveyId").merge(
            train_landcover, on="surveyId")
        self.features = list(train_tab.columns)[1:]
        train_tab = train_tab.fillna(-1).replace(np.inf, -1).replace(-np.inf, -1)

        test_landcover = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/LandCover/GLC24-PA-test-landcover.csv")
        test_solidgrids = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/SoilGrids/GLC24-PA-test-soilgrids.csv")
        test_humanfootprint = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Human Footprint/GLC24-PA-test-human_footprint.csv")
        test_elevation = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Elevation/GLC24-PA-test-elevation.csv")
        test_climate = pd.read_csv(
            f"{self.root_dir}/{env_raster_path}/Climate/Average 1981-2010/GLC24-PA-test-bioclimatic.csv")
        test_tab = test_climate.merge(
            test_elevation, on="surveyId").merge(
            test_humanfootprint, on="surveyId").merge(
            test_solidgrids, on="surveyId").merge(
            test_landcover, on="surveyId")
        test_tab = test_tab.fillna(-1).replace(np.inf, -1).replace(-np.inf, -1)
        # Load Training metadata
        train_landsat_data_path = f"{self.root_dir}/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-landsat_time_series/"
        train_bioclim_data_path = f"{self.root_dir}/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-bioclimatic_monthly/"
        train_sentinel_data_path = f"{self.root_dir}/PA_Train_SatellitePatches_RGB/pa_train_patches_rgb/"
        train_metadata = pd.read_csv(f"{self.root_dir}/GLC24_PA_metadata_train.csv")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = TrainDataset(train_tab, train_bioclim_data_path, train_landsat_data_path, train_sentinel_data_path,
                               train_metadata, self.features, transform=transform)
        self.train_dataset, self.val_dataset = random_split(dataset,
                                                            [int(len(dataset) * self.train_ratio),
                                                             len(dataset) - int(len(dataset) * self.train_ratio)],
                                                            generator=torch.Generator().manual_seed(self.seed), )
        # Load Test metadata
        test_landsat_data_path = f"{self.root_dir}/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-landsat_time_series/"
        test_bioclim_data_path = f"{self.root_dir}/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-bioclimatic_monthly/"
        test_sentinel_data_path = f"{self.root_dir}/PA_Test_SatellitePatches_RGB/pa_test_patches_rgb/"
        test_metadata = pd.read_csv(f"{self.root_dir}/GLC24_PA_metadata_test.csv")

        self.test_dataset = TestDataset(test_tab, test_bioclim_data_path, test_landsat_data_path,
                                        test_sentinel_data_path,
                                        test_metadata, self.features, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
