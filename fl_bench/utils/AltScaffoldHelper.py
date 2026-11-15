import copy
import torch
from torch.optim import Optimizer
import os
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss


def get_lr_values(optimizer: Optimizer):
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]


class AltScaffoldHelper(object):
    def __init__(self):
        self.cnt = 0
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None

    def init(self, model, exclude_vars: list = []):
        self.c_global = copy.deepcopy(model)
        self.c_local = copy.deepcopy(model)
        # HACK: exclude norm and attn.index layers
        c_init_para = {}
        self.exclude_vars = exclude_vars
        for key, value in model.state_dict().items():
            c_init_para[key] = torch.zeros_like(value)
        self.c_global.load_state_dict(c_init_para)
        self.c_local.load_state_dict(c_init_para)

    def get_params(self):
        self.cnt = 0
        c_global_para = self.c_global.state_dict()
        c_local_para = self.c_local.state_dict()
        return c_global_para, c_local_para

    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        net_para = model.state_dict()
        for key in net_para:
            if any(var in key for var in self.exclude_vars):
                continue
            else:
                net_para[key] = net_para[key] - curr_lr * (c_global_para[key] - c_local_para[key])
        model.load_state_dict(net_para)

        self.cnt += 1

    def terms_update(self, model, curr_lr, c_global_para, c_local_para, model_global):

        c_new_para = self.c_local.state_dict()
        self.c_delta_para = copy.deepcopy(self.c_local.state_dict())
        global_model_para = model_global.state_dict()
        net_para = model.state_dict()
        for key in net_para:
            if any(var in key for var in self.exclude_vars):
                c_new_para[key] = torch.zeros_like(c_new_para[key])
            else:
                c_new_para[key] = (
                    c_new_para[key]
                    - c_global_para[key]
                    + (global_model_para[key] - net_para[key]) / (self.cnt * curr_lr)
                )
            self.c_delta_para[key] = (c_new_para[key] - c_local_para[key]).cpu().numpy()
        self.c_local.load_state_dict(c_new_para)

    def load_global_controls(self, weights):
        self.c_global.load_state_dict(weights)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")
        return self.c_delta_para


class DittoHelper(object):
    def __init__(
        self, criterion, model, optimizer, device, app_dir: str, ditto_lambda: float = 0.1, model_epochs: int = 1
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_epochs = model_epochs
        self.prox_criterion = PTFedProxLoss(mu=ditto_lambda)
        self.epoch_global = 0
        self.epoch_of_start_time = 0
        self.best_metric: int = 0
        self.model_file_path = os.path.join(app_dir, "personalized_model.pt")
        self.best_model_file_path = os.path.join(app_dir, "best_personalized_model.pt")

    def init(self, model, exclude_vars: list = []):
        # self.criterion = criterion
        self.model = copy.deepcopy(model)
        # self.optimizer = optimizer

    def load_model(self, global_weights):
        if os.path.exists(self.model_file_path):
            model_data = torch.load(self.model_file_path)
            self.model.load_state_dict(model_data["model"])
            self.epoch_of_start_time = model_data["epoch"]
        else:
            self.model.load_state_dict(global_weights)
            self.epoch_of_start_time = 0
        if os.path.exists(self.best_model_file_path):
            model_data = torch.load(self.best_model_file_path)
            self.best_metric = model_data["best_metric"]

    def save_model(self, is_best=False):
        # save personalized model locally
        model_weights = self.model.state_dict()
        save_dict = {"model": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_metric": self.best_metric})
            torch.save(save_dict, self.best_model_file_path)
        else:
            torch.save(save_dict, self.model_file_path)

    def update_metric_save_model(self, metric):
        self.save_model(is_best=False)
        if metric > self.best_metric:
            self.best_metric = metric
            self.save_model(is_best=True)

    def local_train(self, train_loader, model_global, abort_signal, writer):
        for epoch in range(self.model_epochs):
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch + 1
            for i, batch_data in enumerate(train_loader):
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_ditto = self.prox_criterion(self.model, model_global)
                loss += loss_ditto
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                writer.add_scalar("train_loss_ditto", loss.item(), current_step)
