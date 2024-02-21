import json
import numpy as np
import os
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader
from fl_bench.utils.misc import seed_everything
from monai.data.dataset import Dataset, CacheDataset
from monai.data.decathlon_datalist import load_decathlon_datalist
import monai.transforms as T
from monai.transforms.compose import Compose

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ChestDataModule():
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
        self.batch_size = batch_size
        self.cache_rate = cache_rate

        # transforms:
        self.transform_train = Compose([
            T.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            T.ScaleIntensityd(keys=["image"]),
            T.HistogramNormalized(keys=["image"], min=0, max=1),
            T.Resized(keys=["image"], spatial_size=[224, 224]),
            T.RandRotated(keys=["image"], range_x=np.pi/18, prob=0.5),
            T.RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            T.RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
            T.EnsureTyped(keys=["image", "label"]),
        ])

        self.transform_valid = Compose([
            T.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            T.ScaleIntensityd(keys=["image"]),
            T.HistogramNormalized(keys=["image"], min=0, max=1),
            T.Resized(keys=["image"], spatial_size=[224, 224]),
            T.EnsureTyped(keys=["image", "label"]),
        ])

        dataset_dir = os.path.join(data_dir, "images")
        datalist_paths = os.path.join(data_dir, "datalists")

        if central:
            client_datalist = os.path.join(datalist_paths, "central.json")
        else:
            client_datalist = os.path.join(datalist_paths, client_idx + ".json")

        # self.train_list = load_decathlon_datalist(
        #     data_list_file_path=client_datalist,
        #     is_segmentation=False,
        #     data_list_key="training",
        #     base_dir=dataset_dir
        # )
        # self.valid_list = load_decathlon_datalist(
        #     data_list_file_path=client_datalist,
        #     is_segmentation=False,
        #     data_list_key="validation",
        #     base_dir=dataset_dir
        # )
        self.train_list = pd.read_json(client_datalist)["training"].to_list()
        self.valid_list = pd.read_json(client_datalist)["validation"].to_list()

        self.setup()

    def setup(self):

        if self.cache_rate > 0.0:
            self.train_dataset = CacheDataset(
                data=self.train_list,
                transform=self.transform_train,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
            self.valid_dataset = CacheDataset(
                data=self.valid_list,
                transform=self.transform_valid,
                cache_rate=self.cache_rate,
                num_workers=4,
            )
        else:
            self.train_dataset = Dataset(data=self.train_list, transform=self.transform_train)
            self.valid_dataset = Dataset(data=self.valid_list, transform=self.transform_valid)

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
            num_workers=2
        )

    def teardown(self):
        self.train_dataset = {}
        self.val_dataset = {}
        self.train_dataloader = {}
        self.val_dataloader = {}


class ChestSplitter(FLComponent):
    def __init__(
        self,
        dataset: str,
        data_dir: str,
        split_dir: str = "datalists",
        num_sites: int = 6,
        seed: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.num_sites = num_sites
        self.seed = seed

        seed_everything(seed)

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            # if the split directory with this seed split_dir_seed={seed} already exists, skip the split
            seeded_split_path = os.path.join(self.data_dir, self.split_dir, f"seed={self.seed}")
            if os.path.isdir(seeded_split_path):
                self.log_info(
                    fl_ctx,
                    f"Split directory {seeded_split_path} already exists, skipping the split.",
                )
                return

            self.split(fl_ctx)

    def split(self, fl_ctx):

        self.log_info(
            fl_ctx,
            f"Partitioning {self.dataset} training set into {self.num_sites} with seed = {self.seed}",
        )
        class_sum = self._partition_data()

        os.makedirs(f"{self.split_dir}_seed={self.seed}", exist_ok=True)
        sum_file_name = os.path.join(f"{self.split_dir}_seed={self.seed}", "summary.txt")
        with open(sum_file_name, "w") as sum_file:
            sum_file.write(f"Number of clients: {self.num_sites} \n")
            sum_file.write(f"Seed: {self.seed} \n")
            sum_file.write("Class counts for each client: \n")
            sum_file.write(json.dumps(class_sum))

    def _partition_data(self):

        if self.dataset == "CXR14":
            # load the data csv
            data = pd.read_csv(
                os.path.join(self.data_dir, "Data_Entry_2017_v2020.csv"),
                usecols=["Image Index", "Patient ID", "Finding Labels"],
            )
            # encode the labels and filter out all other columns
            labels = data["Finding Labels"].str.get_dummies(sep="|")
            data = pd.concat([data, labels], axis=1)
            data.drop(columns=["Finding Labels"], inplace=True)

            # filter the training/test rows
            train_list = pd.read_csv(os.path.join(self.data_dir, "train_val_list.txt"), header=None)[0].to_list()
            train_data = data[data['Image Index'].isin(train_list)]
            valid_list = pd.read_csv(os.path.join(self.data_dir, "test_list.txt"), header=None)[0].to_list()
            valid_data = data[data['Image Index'].isin(valid_list)]

            # convert the relative paths to absolute paths
            train_data["Image Index"] = train_data["Image Index"].apply(
                lambda x: os.path.join(self.data_dir, "images", x)
            )
            valid_data["Image Index"] = valid_data["Image Index"].apply(
                lambda x: os.path.join(self.data_dir, "images", x)
            )

            # get the unique patient IDs for the training set
            patients = train_data["Patient ID"].unique()
            # shuffle the patients
            np.random.shuffle(patients)
            # split the patients into self.num_sites
            site_idx = np.array_split(patients, self.num_sites)

            # prepare the decathlon datalist json files for each site
            # with their own training set and the commmon validation set
            for i, site in enumerate(site_idx):
                site_list = train_data[train_data["Patient ID"].isin(site)]
                json_data = {"training": [], "validation": []}
                valid_list = valid_data
                for idx, row in site_list.iterrows():
                    new_item = {}
                    new_item["image"] = row["Image Index"]
                    new_item["label"] = row.iloc[2:].values.tolist()
                    json_data["training"].append(new_item)
                for idx, row in valid_list.iterrows():
                    new_item = {}
                    new_item["image"] = row["Image Index"]
                    new_item["label"] = row.iloc[2:].values.tolist()
                    json_data["validation"].append(new_item)
                with open(f"site-{i+1}.json", "w") as f:
                    json.dump(json_data, f, indent=4)

            # get the class counts for each site
            class_sum = {}
            for i, site in enumerate(site_idx):
                class_sum[i] = data[data["Patient ID"].isin(site)].iloc[:, 12:].sum().to_dict()
                # add the number of samples to the summary, which is the number
                # of patients multiplied by the number of images per patient
                cnt = 0
                for patient in site:
                    cnt += len(data[data["Patient ID"] == patient])
                class_sum[i]["samples"] = cnt

            return class_sum
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
