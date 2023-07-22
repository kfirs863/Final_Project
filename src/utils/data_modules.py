import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.custom_datasets import CustomImageDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        # You can add download or data preparation logic here
        pass

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomImageDataset(self.config['data']['train_data_path'], transform=self.transform)
            self.val_dataset = CustomImageDataset(self.config['data']['val_data_path'], transform=self.transform)

        # Add a condition for 'test' stage
        if stage == 'test' or stage is None:
            self.test_dataset = CustomImageDataset(self.config['data']['test_data_path'], transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'], num_workers=self.config['data']['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'], num_workers=self.config['data']['num_workers'])

    # Add a method for the test dataloader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['data']['batch_size'], num_workers=self.config['data']['num_workers'])
