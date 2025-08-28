import datetime
import time

import numpy as np
import torch
from hy2dl.datasetzoo import get_dataset
from hy2dl.modelzoo import get_model
from hy2dl.training.loss import loss_nll
from hy2dl.utils.config import Config
from hy2dl.utils.optimizer import Optimizer
from hy2dl.utils.utils import set_random_seed, upload_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

# # # PART 1: Initialize information
# Path to .yml file where the experiment settings are stored. The experimet settings can also be defined manually as a dictionary.
path_experiment_settings = "examples/mdn.yml"

# Read experiment settings
config = Config(path_experiment_settings)
config.dump()

# # # PART 2: Create datasets and dataloaders used to train/validate model
# Get dataset class
Dataset = get_dataset(config)

# Dataset training
config.logger.info(f"Loading training data from {config.dataset} dataset")
total_time = time.time()

training_dataset = Dataset(cfg=config, 
                           time_period="training")

training_dataset.calculate_basin_std()
training_dataset.calculate_global_statistics(save_scaler=True)
training_dataset.standardize_data()

config.logger.info(f"Number of entities with valid samples: {len(training_dataset.df_ts)}")
config.logger.info(f"Time required to process {len(training_dataset.df_ts)} entities: {datetime.timedelta(seconds=int(time.time()-total_time))}")
config.logger.info(f"Number of valid training samples: {len(training_dataset)}\n")

# Dataloader training
train_loader = DataLoader(dataset=training_dataset,
                          batch_size=config.batch_size_training,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=training_dataset.collate_fn,
                          num_workers=config.num_workers)

# Print details of a loader´s sample to check that the format is correct
config.logger.info("Details training dataloader".center(60, "-"))
config.logger.info(f"Batch structure (number of batches: {len(train_loader)})")
config.logger.info(f"{'Key':^30}|{'Shape':^30}")

# Loop through the sample dictionary and print the shape of each element
for key, value in next(iter(train_loader)).items():
    if key.startswith(("x_d", "x_conceptual")):
        config.logger.info(f"{key}")
        for i, v in value.items():
            config.logger.info(f"{i:^30}|{str(v.shape):^30}")
    else:
        config.logger.info(f"{key:<30}|{str(value.shape):^30}")
        
config.logger.info("")  # prints a blank line
config.logger.info(f"Loading validation data from {config.dataset} dataset")

# Validate on random basins
if config.validate_n_random_basins > 0:
    entities_ids = np.loadtxt(config.path_entities, dtype="str").tolist()
    random_basins = np.random.choice(entities_ids, size=config.validate_n_random_basins, replace=False).tolist()

total_time = time.time()
validation_dataset = Dataset(cfg=config,
                      time_period="validation",
                      entities_ids=random_basins if config.validate_n_random_basins > 0 else None)
    
validation_dataset.scaler = training_dataset.scaler
validation_dataset.standardize_data()

config.logger.info(f"Time required to process {len(validation_dataset.df_ts)} entities: {datetime.timedelta(seconds=int(time.time()-total_time))}\n")
config.logger.info(f"Number of validation samples: {len(validation_dataset)}\n")

validation_loader = DataLoader(dataset=validation_dataset,
                                  batch_size=config.batch_size_evaluation,
                                  shuffle=False,
                                  drop_last=False,
                                  collate_fn=validation_dataset.collate_fn,
                                  num_workers=config.num_workers)

# # # PART 3: Train model
# Initialize model
set_random_seed(cfg=config)
model = get_model(config).to(config.device)

# Initialize model
set_random_seed(cfg=config)
model = get_model(config).to(config.device)

# Initialize optimizer
optimizer = Optimizer(cfg=config, model=model) 

# Training report structure
config.logger.info("Training model".center(60, "-"))
config.logger.info(f"{'':^16}|{'Trainining':^21}|{'Validation':^21}|")
config.logger.info(f"{'Epoch':^5}|{'LR':^10}|{'Loss':^10}|{'Time':^10}|{'Metric':^10}|{'Time':^10}|")

total_time = time.time()
# Loop through epochs
for epoch in range(1, config.epochs + 1):
    train_time = time.time()
    loss_evol = []
    # Training -------------------------------------------------------------------------------------------------------
    model.train()
    # Loop through the different batches in the training dataset
    iterator = tqdm(train_loader, 
                    desc=f"Epoch {epoch}/{config.epochs}. Training", 
                    unit="batches", 
                    ascii=True, 
                    leave=False)
    
    for idx, sample in enumerate(iterator):
        # reach maximum iterations per epoch
        if config.max_updates_per_epoch is not None and idx >= config.max_updates_per_epoch:
            break

        sample = upload_to_device(sample, config.device)  # upload tensors to device
        optimizer.optimizer.zero_grad()  # sets gradients to zero
        
        # Forward pass of the model
        pred = model(sample)
        loss = loss_nll(
            params=pred["params"],
            weights=pred["weights"],
            dist=model.distribution,
            y_obs=sample["y_obs"]
        )
        loss = loss.sum()
        
        # Backpropagation (calculate gradients)
        loss.backward()
        
        # Update model parameters (e.g, weights and biases)
        optimizer.clip_grad_and_step(epoch, idx)

        # Keep track of the loss per batch
        loss_evol.append(loss.item())
        iterator.set_postfix({"loss": f"{np.mean(loss_evol):.3f}"})

        # remove elements from cuda to free memory
        del sample, pred
        torch.cuda.empty_cache()

    # training report
    report = f'{epoch:^5}|{optimizer.optimizer.param_groups[0]["lr"]:^10.5f}|{np.mean(loss_evol):^10.3f}|{str(datetime.timedelta(seconds=int(time.time()-train_time))):^10}|'

    # Validation -----------------------------------------------------------------------------------------------------
    if epoch % config.validate_every == 0:
        val_time = time.time()
        model.eval()

        iterator = tqdm(validation_loader, 
                desc=f"Epoch {epoch}/{config.epochs}. Validation", 
                unit="batches", 
                ascii=True, 
                leave=False)
        
        with torch.no_grad():
            loss_evol = []
            for sample in iterator:
                sample = upload_to_device(sample, config.device)
                pred = model(sample)
                loss = loss_nll(
                    params=pred["params"],
                    weights=pred["weights"],
                    dist=model.distribution,
                    y_obs=sample["y_obs"]
                )
                loss = loss.sum()
                loss_evol.append(loss.item())
                iterator.set_postfix({"loss": f"{np.nanmean(loss_evol):.3f}"})

            # average loss validation
            report += f"{np.nanmean(loss_evol):^10.3f}|{str(datetime.timedelta(seconds=int(time.time()-val_time))):^10}|"

    # No validation
    else:
        report += f"{'':^10}|{'':^10}|"
    

    # Print report and save model
    config.logger.info(report)
    torch.save(model.state_dict(), config.path_save_folder / "model" / f"model_epoch_{epoch}")
    # modify learning rate
    optimizer.update_optimizer_lr(epoch=epoch)

# print total training time
config.logger.info(f'Total training time: {datetime.timedelta(seconds=int(time.time()-total_time))}\n')