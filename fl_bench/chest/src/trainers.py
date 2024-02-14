
from copy import deepcopy
from pathlib import Path, PurePath
from typing import Any, Dict, Optional, List

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.utils.fl_context_utils import generate_log_message
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import LogMessageTag

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm

from fl_bench.utils.AltScaffoldHelper import AltScaffoldHelper, DittoHelper, get_lr_values
from lightly.utils.scheduler import CosineWarmupScheduler
from .utils import AGC
from .validator import Validator

__all__ = ["BaseProstate2DTrainer", "PartPrivate2DTrainer", 
           "ScaffoldProstate2DTrainer", "DittoProstate2DTrainer"]


class BaseProstate2DTrainer(FLComponent):
    """
    Base class for trainers for the 2-D prostate dataset.

    Args:
        train_config (Dict): Configuration parameters for training.
        model (Optional[nn.Module]): A reference to the learner's model.

    Attributes:
        fl_logger: The logger for federated learning.
        logger: The logger for training.
        init_lr: The initial learning rate.
        max_steps: The maximum number of training steps.
        use_half_precision: Flag indicating whether to use half precision.
        scaler: The gradient scaler for mixed precision training.
        optimizer_dict: The dictionary of optimizer arguments.
        scheduler: The scheduler for the optimizer.
        grad_clipping_type: The type of gradient clipping.
        grad_clipping_value: The value for gradient clipping.
        current_step: The current training step.
        current_round: The current training round.
        opt: The optimizer.
        opt_state: The optimizer state.
        sch: The scheduler.
        sch_state: The scheduler state.
        fedprox_mu: The mu parameter for FedProx.
        criterion_prox: The criterion for FedProx.
        validator: The validator for evaluating the model.
        criterion: The loss criterion.
        gradient_accumulation_steps: The number of gradient accumulation steps.

    Methods:
        log_info: Logs an information message.
        configure_optimizer: Configures the optimizer.
        get_batch: Retrieves a batch of data from the data loader.
        training_loop: Performs the training loop.
        setup: Sets up the trainer.
        cleanup: Cleans up the trainer.
        save_checkpoint: Saves a checkpoint of the model and optimizer.
        load_checkpoint: Loads a checkpoint of the model and optimizer.
        run: Runs the training process.

    """

    def __init__(self, config, model=None):
        super().__init__()

        # rename FL logger to avoid conflict with tensorboard logger
        self.fl_logger = self.logger
        self.logger = None

        # Reset/Init step/round and opt/sched attributes
        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None

        self.max_steps = config["max_steps"]
        self.steps_or_epochs = config["steps_or_epochs"]

        self.optimizer_dict = config["optimizer_args"]
        self.optim_resume = config["optim_resume"]
        self.lr = config["learning_rate"]
        self.scheduler_type = config["scheduler"]

        self.grad_clipping_type = config["grad_clipping_type"]
        self.grad_clipping_value = config["grad_clipping_value"]
        self.gradient_accumulation_steps = 1
        self.use_half_precision = config["use_half_precision"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)

        self.criterion = torch.nn.BCELoss()
        # other options?

        self.fedprox_mu = config["fedprox_mu"]
        self.criterion_prox = None
        if self.fedprox_mu > 0:
            self.criterion_prox = PTFedProxLoss(mu=self.fedprox_mu)

        self.validator = Validator()

    def log_info(self, fl_ctx, msg, fire_event=True):
        # overload FLComponent log_info method to change logger name
        log_msg = generate_log_message(fl_ctx, msg)
        self.fl_logger.info(log_msg)
        if fire_event:
            self._fire_log_event(
                event_type=EventType.INFO_LOG_AVAILABLE, log_tag=LogMessageTag.INFO, log_msg=log_msg, fl_ctx=fl_ctx
            )

    def configure_optimizer(self):

        # Configure optimizer
        optimizer_class = self.optimizer_dict.pop('class', None)
        self.opt = eval(f'torch.optim.{optimizer_class}')(
            self.model.parameters(),
            lr=self.lr,
            **self.optimizer_dict)
        self.optimizer_dict['class'] = optimizer_class

        # If using AGC, wrap the optimizer
        if self.grad_clipping_type == "agc":
            self.opt = AGC(self.model.parameters(),
                           self.opt, lamda=self.grad_clipping_value,
                           model=self.model
                           )

        # Load optimizer state
        if (self.opt_state is not None) and (self.optim_resume=="resume"):
            self.opt.load_state_dict(self.opt_state)

        # Configure scheduler
        if self.scheduler_type == "onecycle":
            self.sch = torch.optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=self.lr,
                total_steps=self.max_steps,
                anneal_strategy="cos",
                final_div_factor=1e4
            )
        elif self.scheduler_type == "warmcosine":
            self.sch = CosineWarmupScheduler(
                self.opt,
                warmup_epochs=self.max_steps*0.2,
                max_epochs=self.max_steps/self.gradient_accumulation_steps,
                end_value=0.001
            )
        elif self.scheduler_type == "cosine":
            self.sch = CosineAnnealingLR(
                self.opt,
                T_max=self.max_steps,
                eta_min=0.001
            )
        elif self.scheduler_type == "none":
            self.sch = torch.optim.lr_scheduler.ConstantLR(
                self.opt,
                factor=1,
                total_iters=self.max_steps
            )
        else:
            raise ValueError(f"Invalid scheduler type {self.scheduler_type}")

        # Load scheduler state
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

        # if steps_or_epochs is epochs, num_steps given to training_loop
        # is the number of epochs, otherwise it is the number of steps
        # the rest of training_loop treats num_steps as actual number of steps
        # so convert num_steps to the number of steps if necessary
        if self.steps_or_epochs == "epochs":
            num_steps = len(data_loader)
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
                    if self.criterion_prox is not None:
                        loss = loss + self.criterion_prox(self.model, self.global_model) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.grad_clipping_type == "norm":
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clipping_value,
                            norm_type=2.0
                        )

                    # Apply gradient
                    self.scaler.step(self.opt)
                    self.sch.step()
                    self.scaler.update()

                    if self.logger is not None:
                        step = self.current_step / self.gradient_accumulation_steps
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
            # Measure post-train train metric every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Dice/Post-Train Train MAUROC",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )

    def setup(self, model, logger, data_loader, num_steps, abort_signal):

        self.model = model

        # global model setup for aux losses
        self.global_model = deepcopy(model).eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        if self.steps_or_epochs == "epochs":
            self.max_steps = len(data_loader) * num_steps

        self.configure_optimizer()

    def cleanup(self):

        # Save opt & sch states
        self.opt_state = (
            deepcopy(self.opt.state_dict())
            if self.optim_resume == "resume"
            else None
        )
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

    def load_checkpoint(self, path: str, model: nn.Module, fl_ctx=None) -> nn.Module:

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
        fl_ctx=None
    ):

        self.setup(model, logger, data_loader, num_steps, abort_signal)
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1
        self.cleanup()


class PartPrivate2DTrainer(BaseProstate2DTrainer):
    """This class is used to train FL models while keeping part of the model private.
    If the private part is bn layers, the model will be trained with FedBN.
    If the private part is attention layers, the model will be trained with FedAttn.
    If the private part is the head, the model will be trained with FedPer.
    Any layer or list of layers can be chosen as private by passing the correct names in private_layers.
    each entry in private_layers should be a string with the name of the layer to be kept private.
    the interpolation method is also configurable between linear, inverse-linear, constant, and fixbn style.

    """

    def __init__(self, config: Dict, private_layers: List[str]):
        super().__init__(config)
        self.private_layers = private_layers
        self.private_layer_state = None
        self.interp_method = config['interp_method']

    def cleanup(self):

        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch_state = deepcopy(self.sch.state_dict())

        self.private_layers_state = {}
        # head.fc for fedper, bn for fedbn, attn for fedattn
        for k,v in self.model.state_dict().items():
            if any(subname in k for subname in self.private_layers):
                self.private_layer_state[k] = deepcopy(v)
        # Cleanup opt, sch & models
        self.sch = None
        self.opt = None
        self.model = None
        self.logger = None
        self.abort_signal = None
        # Cleanup GPU cache
        torch.cuda.empty_cache()

    def save_checkpoint(self, path, model):
        path = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)
        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
            "private_layer_state": self.private_layer_state
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: nn.Module, fl_ctx) -> nn.Module:
        '''
        Load_checkpoint is written with the learner.model object in mind.
        It was originally only called to restore its weights from the checkpoint in the case of a crash.
        '''
        ckpt = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)
        self.private_layer_state = ckpt.get("private_layer_state", None)

        final_dict = {}
        if self.private_layer_state is not None:
            if self.interp_method is None:
                final_dict = deepcopy(self.private_layer_state)
            elif self.interp_method == 'linear':
                # Shared -> Private
                interp_ratio = self.current_step / self.max_steps
            elif self.interp_method == 'inverse_linear':
                # Private -> Shared
                interp_ratio = 1 - self.current_step / self.max_steps
            elif self.interp_method == 'fixbn':
                if self.current_step < self.max_steps / 2:
                    interp_ratio = 0
                else:
                    interp_ratio = 1
            elif self.interp_method == 'const':
                self.log_info(fl_ctx, f'Round {self.current_round}: interp method is const')
                interp_ratio = 0.5

            for k, v in self.private_layer_state.items():
                shared_model_value = model.state_dict()[k]
                final_dict[k] = v * interp_ratio + shared_model_value * (1 - interp_ratio)

            model.load_state_dict(final_dict, strict=False)

        return model

    def run(self, model, data_loader, num_steps, device= "cuda:0", logger=None,
        abort_signal=None, fl_ctx=None):
        self.setup(model, logger, abort_signal)
        # different from base trainer,
        # forces a load_checkpoint to restore the private layers
        if self.current_round > 0:
            self.model = self.load_checkpoint("models/last.pt", model, fl_ctx)
        # Run training
        self.model.train()
        self.training_loop(data_loader, num_steps)
        self.current_round += 1
        self.cleanup()


class ScaffoldProstate2DTrainer(BaseProstate2DTrainer):

    def __init__(self, config):
        super().__init__(config)
        self.scaffold_helper = AltScaffoldHelper()        

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
                    if self.criterion_prox is not None:
                        loss = loss+ self.criterion_prox(self.model, self.global_model) / self.gradient_accumulation_steps

                # Backward
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()                

                #  Gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.grad_clipping_type == "norm":
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clipping_value,
                            norm_type=2.0
                        )

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
            # Measure post-train train metric every 5 rounds starting from the 5th round
            if self.current_round % 5 == 0 and self.current_round > 4:
                post_train_acc = next(iter(self.validator.run(self.model, data_loader).values()))
                self.logger.add_scalar(
                    "Dice/Post-Train Train MAUROC",
                    post_train_acc,
                    self.current_step / self.gradient_accumulation_steps
                )


class DittoProstate2DTrainer(BaseProstate2DTrainer):

    def __init__(self, config, ditto_lambda, ditto_model_epochs=1):
        super().__init__(config)

        self.ditto_lambda = ditto_lambda
        self.dittp_model_epochs = model_epochs
        ditto_model = eval(self.model_arch + "()").to(self.device)
        ditto_optimizer = optim.Adam(
            ditto_model.parameters(), lr=config["ditto_learning_rate"])
        self.ditto_helper = DittoHelper()


# class SupervisedMonaiProstateDittoLearner():

#     def train_config(self, fl_ctx):


#         self.ditto_helper = SupervisedPTDittoHelper(
#             criterion=DiceLoss(sigmoid=True),
#             model=ditto_model,
#             optimizer=ditto_optimizer,
#             device=self.device,
#             app_dir=app_dir,
#             ditto_lambda=self.config_info["ditto_lambda"],
#             model_epochs=self.ditto_model_epochs,
#         )

#     def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal,):

#         # update local model weights with received weights
#         dxo = from_shareable(shareable)
#         global_weights = dxo.data

#         # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
#         local_var_dict = self.model.state_dict()
#         model_keys = global_weights.keys()
#         for var_name in local_var_dict:
#             if var_name in model_keys:
#                 weights = global_weights[var_name]
#                 try:
#                     # reshape global weights to compute difference later on
#                     global_weights[var_name] = np.reshape(
#                         weights, local_var_dict[var_name].shape
#                     )
#                     # update the local dict
#                     local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
#                 except Exception as e:
#                     raise ValueError(
#                         "Convert weight from {} failed with error: {}".format(
#                             var_name, str(e)
#                         )
#                     )
#         self.model.load_state_dict(local_var_dict)

#         # DITTO: Load Ditto personalized model
#         self.ditto_helper.load_model(local_var_dict)
#         # make a copy of model_global
#         model_global = copy.deepcopy(self.model)
#         for param in model_global.parameters():
#             param.requires_grad = False

#         self.local_train(fl_ctx=fl_ctx,train_loader=self.train_loader,model_global=model_global,abort_signal=abort_signal,)
#         self.epoch_of_start_time += self.aggregation_epochs

#         # local train ditto model
#         self.ditto_helper.local_train(train_loader=self.train_loader,model_global=model_global,abort_signal=abort_signal,writer=self.writer,)

#         metric = self.local_valid(self.ditto_helper.model,self.valid_loader,abort_signal,tb_id="val_metric_per_model",record_epoch=self.ditto_helper.epoch_global,)

#         # save model
#         self.ditto_helper.update_metric_save_model(metric=metric)

#         # compute delta model, global model has the primary key set
#         local_weights = self.model.state_dict()
#         model_diff = {}
#         for name in global_weights:
#             if name not in local_weights:
#                 continue
#             model_diff[name] = np.subtract(
#                 local_weights[name].cpu().numpy(),
#                 global_weights[name],
#                 dtype=np.float32,
#             )
#             if np.any(np.isnan(model_diff[name])):
#                 self.system_panic(f"{name} weights became NaN...", fl_ctx)
#                 return make_reply(ReturnCode.EXECUTION_EXCEPTION)

#         # flush the tb writer
#         self.writer.flush()

#         # build the shareable
#         dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
#         dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

#         self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
#         return dxo.to_shareable()


def get_trainer(method, config, **kwargs):
    if method in ["fedavg", "fedprox", "fedopt"]:
        return BaseProstate2DTrainer(config)
    elif method == "fedbn":
        return PartPrivate2DTrainer(config, private_layers=["bn"])
    elif method == "fedattn":
        return PartPrivate2DTrainer(config, private_layers=["attn", "se"])
    elif method == "fedper":
        return PartPrivate2DTrainer(config, private_layers=["head.fc"])
    elif method == "scaffold":
        trainer = ScaffoldProstate2DTrainer(config)
        exclude_vars = ["num_batches_tracked"]
        trainer.scaffold_helper.init(model=kwargs['model'], exclude_vars=exclude_vars)
        return trainer
    elif method == "ditto":
        trainer = None
        # trainer = DittoProstate2DTrainer(config)
        # trainer.ditto_helper.init(model=kwargs['model'])
        return trainer
    else:
        raise ValueError(f"Non-implemented method {method}")
