from typing import Any, Dict

import torch
from monai.inferers.inferer import SlidingWindowInferer, SimpleInferer
from monai.transforms import Compose, EnsureType, Activations, AsDiscrete
from tqdm import tqdm


class Validator(object):
    def __init__(self):
        self.inferer = SimpleInferer()
        self.transform_post = None
        self.metric = None

    def run(self, model, data_loader):
        model.eval()
        metric_score = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    batch["image"] = batch["image"].to("cuda:0")
                    batch["label"] = batch["label"].to("cuda:0")
                    batch["preds"] = self.inferer(batch["image"], model)
                    batch["preds"] = self.transform_post(batch["preds"])
                    self.metric(batch["preds"], batch["label"])

        metric_score = self.metric.aggregate()
        self.metric.reset()

        return {"mean_auroc": metric_score.item()}
