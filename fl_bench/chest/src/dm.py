import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from fl_bench.utils.misc import seed_everything
from torchvision import transforms as T


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
        seed: int = 0
    ):
        self.seed = seed
        seed_everything(seed)
        self.data_dir = data_dir
        self.client_idx = client_idx
        self.central = central
        self.batch_size = batch_size

        # transforms:
        self.transform_train = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229]),
        ])
        self.transform_valid = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229]),
        ])

        self.setup()

    def setup(self):

        self.train_dataset = FedChestDataset(
            data_dir=self.data_dir,
            center=self.client_idx,
            pooled=self.central,
            transform=self.transform_train
        )
        self.valid_dataset = FedChestDataset(
            data_dir=self.data_dir,
            center=self.client_idx,
            pooled=self.central,
            train=False,
            transform=self.transform_valid
        )
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            num_workers=4, worker_init_fn=seed_worker,
            generator=g
        )
        self.val_dataloader = DataLoader(
            self.valid_dataset, batch_size=10,
            shuffle=False,
            num_workers=4
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}


class ChestXRay14Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img


class FedChestDataset(Dataset):
    def __init__(
        self,
        data_dir,
        center,
        pooled,
        transform=None
    ):
        self.data_dir = data_dir
        self.center = center
        self.pooled = pooled
        self.transform = transform

        self.train_list = []
        self.valid_list = []

        self.setup()

    def setup(self):
        pass

class ChestDataSplitter(FLComponent):
    def __init__(
        self,
        split_dir: str = None,
        num_sites: int = 5,
        alpha: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.alpha = alpha
        self.seed = seed

        seed_everything(seed)

        if alpha < 0.0:
            raise ValueError(f"Alpha should be non-negative but was {alpha}.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.split(fl_ctx)

    def split(self, fl_ctx: FLContext):
        np.random.seed(self.seed)

        self.log_info(
            fl_ctx,
            f"Partitioning CXR14 training set into {self.num_sites} sites with Dirichlet sampling under alpha {self.alpha}",
        )
        site_idx, class_sum = self._partition_data()

        # write to files
        if self.split_dir is None:
            raise ValueError(
                "You need to define a valid `split_dir` when splitting the data."
            )
        if not os.path.isdir(self.split_dir):
            os.makedirs(self.split_dir)
        sum_file_name = os.path.join(self.split_dir, "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Dirichlet sampling parameter: {self.alpha} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

        site_file_path = os.path.join(self.split_dir, "site-")
        for site in range(self.num_sites):
            site_file_name = site_file_path + str(site + 1) + ".npy"
            np.save(site_file_name, np.array(site_idx[site]))

    def load_cxr14_data(self):
        # download data
        train_dataset = CXR14Dataset(
            root=os.getenv("CXR14_ROOT"),
            split="train",
        )
        train_label = np.array(train_dataset.labels)
        return train_label

    def _partition_data(self):
        train_label = self.load_medmnist_data()

        min_size = 0
        K = 11
        N = train_label.shape[0]
        site_idx = {}

 

        return site_idx, class_sum
