import os
import random
from typing import Dict, Union

import numpy as np
import torch


class Optimizer:
    """Manage the optimizer.

    Parameters
    ----------
    model:
        Model to be optimized
    model_configuration : Dict[str, Union[int, float, str, dict]]
        Configuration of the model
    optimizer : str
        name of the optimizer

    """

    def __init__(self, model, model_configuration: Dict[str, Union[int, float, str, dict]], optimizer: str = None):
        # At the time, only Adam is supported
        if optimizer is None or optimizer.lower() == "adam":
            optimizer_class = torch.optim.Adam

        if (  # if learning rate is a float and no scheduler is used
            isinstance(model_configuration.get("learning_rate"), float)
            and "adapt_learning_rate_epoch" not in model_configuration
            and "adapt_gamma_learning_rate" not in model_configuration
        ):
            self.learning_rate = model_configuration.get("learning_rate")
            self.optimizer = optimizer_class(model.parameters(), lr=self.learning_rate)
            self.learning_rate_type = "constant"

        elif (  # if learning rate is a float and a scheduler is used
            isinstance(model_configuration.get("learning_rate"), float)
            and "adapt_learning_rate_epoch" in model_configuration
            and "adapt_gamma_learning_rate" in model_configuration
        ):
            self.learning_rate = model_configuration.get("learning_rate")

            self.optimizer = optimizer_class(model.parameters(), lr=self.learning_rate)
            self.learning_rate_type = "scheduler"

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=model_configuration["adapt_learning_rate_epoch"],
                gamma=model_configuration["adapt_gamma_learning_rate"],
            )

        elif (  # if learning rate is a dictionary with a custom scheduler
            isinstance(model_configuration.get("learning_rate"), Dict)
        ):
            self.learning_rate = model_configuration.get("learning_rate")
            self.optimizer = optimizer_class(model.parameters(), lr=self._find_learning_rate(epoch=1))
            self.learning_rate_type = "custom_scheduler"

        else:
            # Raise an error if no valid learning rate type is provided
            raise ValueError("Please indicate a valid type of learning rate in the configuration.")

    def _find_learning_rate(self, epoch: int):
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
        """Perform the optimizer step.  
        This involves clipping the gradients with a maximum norm of 1 and updating the 
        otimizer weights.

        """
        # clip gradients to mitigate exploding gradients issues
        try:
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1, error_if_nonfinite=True)
        except Exception as e:
            # if the gradients still explode after norm_clipping, we skip the optimization step
            print(f"Batch {batch} in Epoch {epoch} was skipped during optimization due to gradient instability.")
            print(f"Error: {e}")
            return
        
        # update the optimizer weights
        self.optimizer.step()

        return

def create_folder(folder_path: str):
    """Create a folder to store the results.

    Checks if the folder where one will store the results exist. If it does not, it creates it.

    Parameters
    ----------
    folder_path : str
        Path to the location of the folder

    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def upload_to_device(sample: dict, device: str):
    """Upload the different tensors, contained in a dictionary, to the device.

    Parameters
    ----------
    sample : dict
        Dictionary with the different tensors that will be used for the forward pass.
    device: str
        cpu, gpu

    """
    for key in sample.keys():
        if key not in ("basin", "date"):
            sample[key] = sample[key].to(device)
    return sample


def set_random_seed(seed: int = None):
    """Set a seed for various packages to be able to reproduce the results.

    Parameters
    ----------
    seed : int
        Number of the seed

    """
    if seed is None:
        seed = int(np.random.uniform(low=0, high=1e6))

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def write_report(file_path: str, text: str):
    """Write a given text into a text file.

    If the file where one wants to write does not exists, it creates a new one.

    Parameters
    ----------
    file_path : str
        Path to the file where
    text : str
        Text that wants to be added

    """
    if os.path.exists(file_path):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    highscore = open(file_path, append_write)
    highscore.write(text + "\n")
    highscore.close()
