import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from .config import train_config

def collate_fn(batch):
    return batch


class CategoryDataModule(pl.LightningDataModule):

    def __init__(self, dataset):
        super().__init__()
        trainset_ratio = 0.8
        self.dataset = dataset
        self.trainset, self.valset = random_split(dataset, [trainset_ratio, 1 - trainset_ratio])
        self.testset = dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass


    @property
    def _dataloader_kwargs(self):
        return dict(
            num_workers=4,
            pin_memory=True,
            batch_size=train_config.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self):
        kwargs = self._dataloader_kwargs
        # kwargs['batch_size'] *= 5
        return DataLoader(self.valset, **kwargs)  # type: ignore

    def test_dataloader(self):
        kwargs = self._dataloader_kwargs
        # kwargs['batch_size'] *= 4  # type: ignore
        kwargs['num_workers'] *= 2  # type: ignore
        return DataLoader(self.testset, **kwargs)  # type: ignore
    
    def predict_dataloader(self):
        return DataLoader(self.dataset, **self._dataloader_kwargs)