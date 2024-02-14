import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from fl_bench.utils.misc import seed_everything

from monai.data.dataset import Dataset, CacheDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ChestXRay14DataModule():
    def __init__(
        self,
        data_dir: str,
        client_idx: str,
        central: bool,
        batch_size: int = 64,
        cache_rate: float = 1.0,
        seed: int = 0
    ):
        self.seed = seed
        seed_everything(seed)
        self.data_dir = data_dir
        self.client_idx = client_idx
        self.central = central
        self.batch_size = batch_size
        self.cache_rate = cache_rate

        # transforms:
        self.transform = None


        self.setup()

    def setup(self):

        if self.cache_rate > 0.0:
            self.train_dataset = CacheDataset(
                data=self.train_list,
                transform=self.transform,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
            self.valid_dataset = CacheDataset(
                data=self.valid_list,
                transform=self.transform,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
        else:
            self.train_dataset = Dataset(data=self.train_list, transform=self.transform)
            self.valid_dataset = Dataset(data=self.valid_list, transform=self.transform)

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            num_workers=8, worker_init_fn=seed_worker,
            generator=g
        )
        self.val_dataloader = DataLoader(
            self.valid_dataset, batch_size=1,
            shuffle=False,
            num_workers=2
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}
