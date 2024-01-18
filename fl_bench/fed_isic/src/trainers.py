from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from .validator import Validator
from fl_bench.utils.AltScaffoldHelper import AltScaffoldHelper, get_lr_values
from fl_bench.fed_isic.src.utils import WeightedFocalLoss
from lightly.utils.scheduler import CosineWarmupScheduler

class BaseTrainer(object):
    def __init__(self, train_config: Dict, method: str, class_weights):
        self.init_lr = train_config["learning_rate"]
        self.max_steps = train_config["max_steps"]
        self.use_half_precision = train_config["use_half_precision"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)
        self.method = method
        self.optimizer_dict = train_config["optimizer_args"]
        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None
        self.fedprox_mu = train_config["fedprox_mu"]
        self.criterion_prox = None
        self.validator = Validator()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = WeightedFocalLoss(alpha=class_weights)
        self.gradient_accumulation_steps = 1

    def configure_optimizer(self):
        optimizer_class = self.optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.init_lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        # self.sch = CosineAnnealingLR(self.opt, T_max=self.max_steps/self.gradient_accumulation_steps, eta_min=1e-4)
        # self.sch = torch.optim.lr_scheduler.MultiStepLR(
        #     self.opt,
        #     milestones=[873, 1455, 2037, 2619, 3201, 3783, 4365, 4947],
        #     gamma=0.5
        # )
        self.sch = CosineWarmupScheduler(
            self.opt,
            warmup_epochs=self.max_steps*0.15,
            max_epochs=self.max_steps/self.gradient_accumulation_steps,
            end_value=0.005
        )
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    # self.scaler.unscale_(self.opt)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def setup(self, model, logger, abort_signal):
        self.model = model

        # global model setup for aux losses
        # self.global_model = deepcopy(model).eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)

        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()

class FedBNTrainer(BaseTrainer):
    '''
        Only works with efficientnet-b0 and other nets that name bn layers with "bn"
    '''
    def __init__(self, train_config: Dict, method: str, class_weights):
        super().__init__(train_config, method, class_weights)
        self.bn_state = None

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)
                        self.logger.add_scalar('Batch Norm', self.model.base_model._bn0.running_var[0].item(), step)


                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def configure_optimizer(self):
        optimizer_class = self.optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.init_lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        self.sch = CosineWarmupScheduler(
            self.opt,
            warmup_epochs=self.max_steps*0.05,
            max_epochs=self.max_steps/self.gradient_accumulation_steps,
            end_value=1
        )
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def setup(self, model, logger, abort_signal):
        self.model = model
        
        if self.bn_state is not None:
            self.model.load_state_dict(self.bn_state, strict=False)

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # bn state dict
        self.bn_state = {}
        for k,v in self.model.state_dict().items():
            if 'bn' in k:
                self.bn_state[k] = deepcopy(v)

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
            "bn_state": self.bn_state
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)
        self.bn_state = ckpt.get('bn_state', None)

        model.load_state_dict(ckpt["model"])
        # load batch norm parameters
        if self.bn_state is not None:
            self.model.load_state_dict(self.bn_state, strict=False)
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)
        if self.current_round > 0:
            self.model = self.load_checkpoint("models/last.pt", self.model)
        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()


class FedAttnTrainer(BaseTrainer):
    '''
        Compatible with all timm models that name attention layers with "attn"
    '''
    def __init__(self, train_config: Dict, method: str, class_weights):
        super().__init__(train_config, method, class_weights)
        self.attn_state = None

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)
                        # attn metric goes here


                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def configure_optimizer(self):
        optimizer_class = self.optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.init_lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)

        self.sch = CosineWarmupScheduler(
            self.opt,
            warmup_epochs=self.max_steps*0.2,
            max_epochs=self.max_steps/self.gradient_accumulation_steps,
            end_value=0.001
        )
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def setup(self, model, logger, abort_signal):
        self.model = model
        
        if self.attn_state is not None:
            self.model.load_state_dict(self.attn_state, strict=False)

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # attn state dict
        self.attn_state = {}
        for k,v in self.model.state_dict().items():
            if 'attn' in k:
                self.attn_state[k] = deepcopy(v)

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
            "attn_state": self.attn_state
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)
        self.attn_state = ckpt.get('attn_state', None)

        model.load_state_dict(ckpt["model"])
        # load attention parameters
        if self.attn_state is not None:
            self.model.load_state_dict(self.attn_state, strict=False)
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)
        if self.current_round > 0:
            self.model = self.load_checkpoint("models/last.pt", self.model)
        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()

class ScaffoldTrainer(BaseTrainer):
    def __init__(self, train_config: Dict, method: str, class_weights):
        super().__init__(train_config, method, class_weights)
        self.scaffold_helper = AltScaffoldHelper()

    def setup(self, model, logger, abort_signal):
        self.model = model
        # global model setup for aux losses
        self.global_model = deepcopy(model).eval()
        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        c_global_para, c_local_para = self.scaffold_helper.get_params()

        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")


            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    # self.scaler.unscale_(self.opt)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    # SCAFFOLD model update step
                    curr_lr = get_lr_values(self.opt)[0]
                    self.scaffold_helper.model_update(
                        model=self.model,
                        curr_lr=curr_lr,
                        c_global_para=c_global_para,
                        c_local_para=c_local_para
                    )

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item() 

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # SCAFFOLD terms update step
            self.scaffold_helper.terms_update(
                model=self.model,
                curr_lr=curr_lr,
                c_global_para=c_global_para,
                c_local_para=c_local_para,
                model_global=self.global_model,
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )


class BaseTrainerv2(object):
    """
    Difference from BaseTrainer:
    - gradient clipping (l2 norm at 1.0)
    - slightly different lr scheduler: 0.2 warmup, 0.001 final lr

    """    
    def __init__(self, train_config: Dict, method: str, class_weights):
        self.init_lr = train_config["learning_rate"]
        self.max_steps = train_config["max_steps"]
        self.use_half_precision = train_config["use_half_precision"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)
        self.method = method
        self.optimizer_dict = train_config["optimizer_args"]
        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None
        self.fedprox_mu = train_config["fedprox_mu"]
        self.criterion_prox = None
        self.validator = Validator()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = WeightedFocalLoss(alpha=class_weights)
        self.gradient_accumulation_steps = 1

    def configure_optimizer(self):
        optimizer_class = self.optimizer_dict.pop('class', None)
        assert optimizer_class in ['SGD', 'Adam', 'AdamW']
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.init_lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class
        if self.opt_state is not None:
            self.opt.load_state_dict(self.opt_state)


        self.sch = CosineWarmupScheduler(
            self.opt,
            warmup_epochs=self.max_steps*0.2,
            max_epochs=self.max_steps/self.gradient_accumulation_steps,
            end_value=0.001
        )
        if self.sch_state is not None:
            self.sch.load_state_dict(self.sch_state)

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")

            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item()

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def setup(self, model, logger, abort_signal):
        self.model = model

        # global model setup for aux losses
        # self.global_model = deepcopy(model).eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def cleanup(self):
        # Save opt & sch states
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None

        self.logger = None
        self.abort_signal = None

        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path: str, model: nn.Module) -> None:
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module) -> nn.Module:
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model

    def run(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_steps: int,
        device: str = "cuda:0",
        logger: Optional[SummaryWriter] = None,
        abort_signal: Optional[Any] = None,
    ):
        self.setup(model, logger, abort_signal)

        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()



class ScaffoldTrainerv2(BaseTrainerv2):
    def __init__(self, train_config: Dict, method: str, class_weights):
        super().__init__(train_config, method, class_weights)
        self.scaffold_helper = AltScaffoldHelper()

    def setup(self, model, logger, abort_signal):
        self.model = model
        # global model setup for aux losses
        self.global_model = deepcopy(model).eval()
        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda:0"):
        target_step = self.current_step + num_steps
        avg_loss = 0.0
        c_global_para, c_local_para = self.scaffold_helper.get_params()

        with tqdm(total=num_steps, dynamic_ncols=True) as pbar:
            # Configure progress bar
            pbar.set_description(f"Round {self.current_round}")


            for i, batch in enumerate(self.get_batch(data_loader, num_steps)):
                # Forward
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    preds = self.model(image)
                    loss = self.criterion(preds, label) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    # SCAFFOLD model update step
                    curr_lr = get_lr_values(self.opt)[0]
                    self.scaffold_helper.model_update(
                        model=self.model,
                        curr_lr=curr_lr,
                        c_global_para=c_global_para,
                        c_local_para=c_local_para
                    )

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
                        self.logger.add_scalar("Loss/Batch Loss", loss, step)
                        self.logger.add_scalar("Learning Rate", self.sch.get_last_lr()[-1], step)

                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.2f}")
                pbar.update(1)

                # Write average loss to tensorboard
                avg_loss += loss.item() 

                self.current_step += 1
                if self.current_step >= target_step:
                    break
                if self.abort is not None and self.abort.triggered:
                    break

            self.logger.add_scalar(
                "Loss/Round Loss",
                avg_loss / num_steps,
                self.current_step / self.gradient_accumulation_steps
            )
            # SCAFFOLD terms update step
            self.scaffold_helper.terms_update(
                model=self.model,
                curr_lr=curr_lr,
                c_global_para=c_global_para,
                c_local_para=c_local_para,
                model_global=self.global_model,
            )
            # Post-train accuracy every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Accuracy/Post-Train Train Accuracy",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )
