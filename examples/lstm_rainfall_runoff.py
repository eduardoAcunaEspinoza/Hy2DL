# Import necessary packages
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import xarray as xr

from hy2dl.datasetzoo import get_dataset
from hy2dl.evaluation import calculate_metrics, get_tester
from hy2dl.modelzoo import get_model
from hy2dl.training.basetrainer import BaseTrainer
from hy2dl.utils.config import Config

os.chdir(sys.path[0])  # Change working directory to the script's location
base_dir = Path.cwd().resolve()
color_palette = {"observed": "#377eb8", "simulated": "#4daf4a"}


# ------------------------------
# Part 1. Initialize information
# ------------------------------
if __name__ == "__main__":
    # Read experiment settings
    path_experiment_settings = "../examples/configs/camels_gb.yml"
    config = Config(path_experiment_settings, base_dir=base_dir)
    config.init_experiment()
    config.dump()

    Dataset = get_dataset(config)
    Tester = get_tester(config)

    # Create training dataset
    training_dataset = Dataset(cfg=config, time_period="training")
    training_dataset.setup_dataset()
    # Initialize training object
    trainer = BaseTrainer(cfg=config, training_dataset=training_dataset)

    validation_dataset = Dataset(cfg=config, time_period="validation")
    validation_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")
    tester_validation = Tester(cfg=config, evaluation_dataset=validation_dataset)

    # Training report structure
    validation_headers = "".join([f"{m:^10}|" for m in config.validation_metric])
    config.logger.info("Training model".center(60, "-"))
    config.logger.info(f"{'':^16}|{'Training':^21}|{'Validation':^{(11 * len(config.validation_metric)) + 10}}|")
    config.logger.info(f"{'Epoch':^5}|{'LR':^10}|{'Loss':^10}|{'Time':^10}|{validation_headers}{'Time':^10}|")

    # Loop through epochs
    total_time = time.time()
    for epoch in range(1, config.epochs + 1):
        trainer.train_model(epoch=epoch)  # Training
        tester_validation.validate_model(model=trainer.model, epoch=epoch)  # Validation
        config.logger.info(trainer.report + tester_validation.validation_report)  # report

    config.logger.info(f"Total training time: {datetime.timedelta(seconds=int(time.time() - total_time))}\n")
    shutil.rmtree(tester_validation.path_zarr, ignore_errors=True)  # delete validation results

    # If I already trained a model, I can re-construct it using the saved parameters from a given epoch
    model = get_model(config).to(config.device)
    model.load_state_dict(
        torch.load(config.path_save_folder / "model" / f"model_epoch_{config.epochs}", map_location=config.device)
    )

    testing_dataset = Dataset(cfg=config, time_period="testing")
    testing_dataset.setup_dataset(check_nan=False, path_scaler=config.path_save_folder / "scaler.yml")
    tester_testing = Tester(cfg=config, evaluation_dataset=testing_dataset)

    config.logger.info("Testing model...")
    testing_time = time.time()
    tester_testing.evaluate_model(model=model)
    config.logger.info("Testing completed.")
    config.logger.info(f"Total testing time: {datetime.timedelta(seconds=int(time.time() - testing_time))}\n")

    test_results = xr.open_zarr(tester_testing.path_zarr)
    testing_metrics = calculate_metrics(
        ds_results=test_results, metric_name=config.testing_metrics, distribution=config.distribution
    )
    testing_metrics.to_zarr(config.path_save_folder / "testing_metrics.zarr", mode="w")
