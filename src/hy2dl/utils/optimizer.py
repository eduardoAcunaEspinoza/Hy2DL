import warnings

import torch
from hy2dl.utils.config import Config


class Optimizer:
    """Manage the optimizer.

    Parameters
    ----------
    cfg : Config
        Configuration file.
    model:
        Model to be optimized
    """

    def __init__(self, cfg: Config, model: torch.nn.Module):
        # At this time, only Adam is supported
        if cfg.optimizer == "adam":
            optimizer_class = torch.optim.Adam
        else:
            raise ValueError(f"Optimizer {cfg.optimizer} is not supported. Only 'adam' is currently available.")

        if (  # if learning rate is a float and no scheduler is used
            isinstance(cfg.learning_rate, float) and cfg.steplr_step_size is None and cfg.steplr_gamma is None
        ):
            self.learning_rate_type = "constant"
            self.learning_rate = cfg.learning_rate
            self.optimizer = optimizer_class(model.parameters(), lr=self.learning_rate)

        elif (  # if learning rate is a float and a information about scheduler is provided
            isinstance(cfg.learning_rate, float)
            and isinstance(cfg.steplr_step_size, int)
            and isinstance(cfg.steplr_gamma, float)
        ):
            self.learning_rate_type = "scheduler"
            self.learning_rate = cfg.learning_rate
            self.optimizer = optimizer_class(model.parameters(), lr=self.learning_rate)

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.steplr_step_size,
                gamma=cfg.steplr_gamma,
            )

        elif (  # if learning rate is a dictionary with a custom scheduler
            isinstance(cfg.learning_rate, dict)
        ):
            self.learning_rate_type = "custom_scheduler"
            self.learning_rate = cfg.learning_rate
            self.optimizer = optimizer_class(model.parameters(), lr=self._find_learning_rate(epoch=1))

        else:
            # Raise an error if no valid learning rate type is provided
            raise ValueError("Please indicate a valid type of learning rate in the configuration.")

    def _find_learning_rate(self, epoch: int) -> float:
        """Return learning rate for a given epoch, based on a custom scheduler.

        Parameters
        ----------
        epoch: int
            Epoch for which the learning rate is needed

        Returns
        -------
        float
            learning rate for the given epoch, determined by the custom scheduler

        """
        sorted_keys = sorted(self.learning_rate.keys(), reverse=True)
        for key in sorted_keys:
            if epoch >= key:
                return self.learning_rate[key]
        raise ValueError(f"No valid learning rate found for epoch {epoch}. Check your scheduler keys.")

    def update_optimizer_lr(self, epoch: int):
        """Update the learning rate

        Parameters
        ----------
        epoch : int
            Current epoch

        """
        if self.learning_rate_type == "scheduler":
            self.scheduler.step()
        elif self.learning_rate_type == "custom_scheduler":
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._find_learning_rate(epoch=epoch)

    def clip_grad_and_step(self, epoch: int, batch: int) -> None:
        """Perform an optimization step.

        Before performing a step with the optimizer, tries to clip the gradients with a maximum norm of 1.

        Parameters
        ----------
        epoch : int
            Current epoch
        batch : int
            Batch ID of the current batch

        """
        # clip gradients to mitigate exploding gradients issues
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.optimizer.param_groups[0]["params"], max_norm=1, error_if_nonfinite=True
            )
        except RuntimeError as e:
            # if the gradients still explode after norm_clipping, we skip the optimization step
            warnings.warn(
                (
                    f"Batch {batch} in Epoch {epoch} was skipped during optimization "
                    f"due to gradient instability. Error:\n{e}"
                ),
                stacklevel=2,
            )

            return

        # update the optimizer weights
        self.optimizer.step()

        return
