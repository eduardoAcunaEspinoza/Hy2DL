import datetime
import time
from typing import Any

import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from tqdm import tqdm

from hy2dl.datasetzoo import BaseDataset
from hy2dl.modelzoo import get_model
from hy2dl.training import get_loss
from hy2dl.training.optimizer import Optimizer
from hy2dl.utils.config import Config
from hy2dl.utils.utils import set_random_seed, upload_to_device


class BaseTrainer(object):
    """Class to train a model

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    training_dataset : hy2dl.datasetzoo.basedataset.BaseDataset
        Dataset used for training

    """

    def __init__(self, cfg: Config, training_dataset: BaseDataset):
        set_random_seed(cfg=cfg)

        self.cfg = cfg

        # Initialize model and loss
        self.model = get_model(self.cfg).to(self.cfg.device)
        self.loss = get_loss(cfg=self.cfg)
        # Initialize optimizer
        self.optimizer = Optimizer(cfg=self.cfg, model=self.model)

        # Create training dataloader
        self.train_loader = DataLoader(
            dataset=training_dataset,
            batch_size=self.cfg.batch_size_training,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            collate_fn=training_dataset.collate_fn,
            worker_init_fn=training_dataset.dask_worker_init_fn,
        )
        self._check_samples()
        self.report = None

    def _check_samples(self):
        # Print details of a loader´s sample to check that the format is correct
        self.cfg.logger.info("Details training dataloader".center(60, "-"))
        self.cfg.logger.info(f"Batch structure (number of batches: {len(self.train_loader)})")
        self.cfg.logger.info(f"{'Key':^30}|{'Shape':^30}")

        # Auxiliary printing function
        def _log_item(name: str, val: Any, indent: int = 0):
            prefix = "  " * indent
            display_name = f"{prefix}{name}"

            if isinstance(val, dict):
                self.cfg.logger.info(f"{display_name:<30}|{'':^30}")
                for sub_key, sub_val in val.items():
                    _log_item(sub_key, sub_val, indent=indent + 1)
            elif hasattr(val, "shape"):
                self.cfg.logger.info(f"{display_name:<30}|{str(val.shape):^30}")
            elif val is None:
                self.cfg.logger.info(f"{display_name:<30}|{'None':^30}")
            else:
                self.cfg.logger.info(f"{display_name:<30}|{type(val).__name__:^30}")

        # Load sample and print structure
        sample = next(iter(self.train_loader))
        for key, value in sample.items():
            _log_item(key, value)
        self.cfg.logger.info("")

    def train_model(self, epoch):
        self.model.train()
        iterator = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.epochs}. Training",
            unit="batches",
            ascii=True,
            leave=False,
        )
        # Loop through the batches
        running_loss = 0.0
        training_time = time.time()
        with threadpool_limits(limits=1):
            for idx, sample in enumerate(iterator):
                if self.cfg.max_updates_per_epoch is not None and idx >= self.cfg.max_updates_per_epoch:
                    break  # reach maximum iterations per epoch

                sample = upload_to_device(sample, self.cfg.device)  # upload tensors to device
                self.optimizer.optimizer.zero_grad()  # sets gradients to zero

                # Inject noise if noise_level is specified in the config
                if self.cfg.noise_level is not None:
                    sample = self._inject_noise(sample=sample, noise_level=self.cfg.noise_level)

                # Forward pass of the model
                pred = self.model(sample)

                loss = self.loss(pred=pred, sample=sample)  # calcuate loss
                loss.backward()  # backpropagation (calculate gradients)
                self.optimizer.clip_grad_and_step(epoch, idx)  # update model parameters (e.g, weights and biases)

                # Keep track of the loss evolution
                running_loss += (loss.detach().item() - running_loss) / (idx + 1)
                iterator.set_postfix({"average loss": f"{running_loss:.3f}"})

                # remove elements from cuda to free memory
                del sample, pred

        report_lr = self.optimizer.optimizer.param_groups[0]["lr"]
        report_time = str(datetime.timedelta(seconds=int(time.time() - training_time)))
        self.report = f"{epoch:^5}|{report_lr:^10.5f}|{running_loss:^10.3f}|{report_time:^10}|"
        self.optimizer.update_optimizer_lr(epoch=epoch + 1)
        torch.save(self.model.state_dict(), self.cfg.path_save_folder / "model" / f"model_epoch_{epoch}")

    def _inject_noise(self, sample: dict[str, Any], noise_level: float) -> dict[str, Any]:
        def _add_noise(v):
            if isinstance(v, torch.Tensor):
                return v * (1.0 + torch.randn_like(v) * noise_level)
            elif isinstance(v, dict):
                return {k: _add_noise(val) for k, val in v.items()}
            return v

        sample["y_obs"] = sample["y_obs"] * (1.0 + torch.randn_like(sample["y_obs"]) * noise_level)
        for key, value in sample.items():
            if key.startswith("x_d"):
                sample[key] = _add_noise(value)
        return sample
