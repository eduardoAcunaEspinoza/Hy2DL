import argparse
import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from hy2dl.datasetzoo import get_dataset
from hy2dl.evaluation.metrics import calc_nse
from hy2dl.modelzoo import get_model
from hy2dl.training.loss import loss_nll
from hy2dl.utils.config import Config
from hy2dl.utils.distributions import Distribution
from hy2dl.utils.utils import upload_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def to_xarray(out_dict):
    # Get all basin IDs
    basin_ids = list(out_dict.keys())
    
    # Get dates from first basin (assuming all basins have same dates)
    dates = out_dict[basin_ids[0]]['date']
    num_days = dates.shape[0]
    
    # Get dimensions from the first basin
    first_basin = out_dict[basin_ids[0]]
    num_basins = len(basin_ids)
    predict_last_n = first_basin['y_obs'].shape[1]
    num_targets = first_basin['y_obs'].shape[2]
    num_samples = first_basin['y_hat'].shape[2]
    
    # Get parameter names dynamically from params subdictionary
    param_names = list(first_basin['params'].keys())
    num_components = first_basin['params'][param_names[0]].shape[2]
    
    # Initialize arrays for fixed variables
    y_obs_array = np.zeros((num_basins, num_days, predict_last_n, num_targets))
    y_hat_array = np.zeros((num_basins, num_days, predict_last_n, num_samples, num_targets))
    weights_array = np.zeros((num_basins, num_days, predict_last_n, num_components, num_targets))
    
    # Initialize arrays for parameter variables dynamically
    param_arrays = {}
    for param_name in param_names:
        param_arrays[param_name] = np.zeros((num_basins, num_days, predict_last_n, num_components, num_targets))
    
    # Fill arrays
    for idx, basin_id in enumerate(basin_ids):
        basin_data = out_dict[basin_id]

        y_obs_array[idx] = basin_data['y_obs']
        y_hat_array[idx] = basin_data['y_hat']
        weights_array[idx] = basin_data['weights']
        
        # Fill parameter arrays dynamically
        for param_name in param_names:
            param_arrays[param_name][idx] = basin_data['params'][param_name]

    # Create coordinate arrays
    coords = {
        'basin_id': basin_ids,
        'date': dates[:, -1],
        'predict_last_n': np.arange(predict_last_n),
        'num_targets': np.arange(num_targets),
        'num_samples': np.arange(num_samples),
        'num_components': np.arange(num_components)
    }
    
    # Create data variables
    data_vars = {
        'y_obs': (['basin_id', 'date', 'predict_last_n', 'num_targets'], y_obs_array),
        'y_hat': (['basin_id', 'date', 'predict_last_n', 'num_samples', 'num_targets'], y_hat_array),
        'weights': (['basin_id', 'date', 'predict_last_n', 'num_components', 'num_targets'], weights_array)
    }
    
    # Add parameter variables dynamically
    for param_name in param_names:
        data_vars[param_name] = (['basin_id', 'date', 'predict_last_n', 'num_components', 'num_targets'], param_arrays[param_name])
    
    # Create the dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add attributes for better documentation
    ds.attrs['description'] = 'Basin prediction data with observations, predictions, and model parameters'
    ds['y_obs'].attrs['description'] = 'Observed values'
    ds['y_hat'].attrs['description'] = 'Predicted values (ensemble samples)'
    ds['weights'].attrs['description'] = 'Weight values'
    
    # Add parameter descriptions dynamically
    for param_name in param_names:
        ds[param_name].attrs['description'] = f'{param_name.capitalize()} parameters'
    
    return ds

def generate_netcdf(cfg: Config, path_model: Path):
    path_experiment = path_model.parent.parent

    # Load model
    model = get_model(cfg).to(cfg.device)
    model.load_state_dict(torch.load(path_model, map_location=cfg.device))

    # Read Scaler
    with open(path_experiment / "scaler.pickle", "rb") as file:
        scaler = pickle.load(file)
    
    # Get dataset class
    Dataset = get_dataset(config)

    # In evaluation (validation and testing) we will create an individual dataset per basin
    config.logger.info(f"Loading testing data from {config.dataset} dataset")

    entities_ids = np.loadtxt(config.path_entities_testing, dtype="str").tolist()
    iterator = tqdm([entities_ids] if isinstance(entities_ids, str) else entities_ids, 
                    desc="Processing entities", 
                    unit="entity", 
                    ascii=True)

    total_time = time.time()
    testing_dataset = {}
    for entity in iterator:
        dataset = Dataset(cfg= config, 
                        time_period= "testing",
                        check_NaN=False,
                        entities_ids=entity)

        dataset.scaler = scaler
        dataset.standardize_data(standardize_output=False)
        testing_dataset[entity] = dataset

    config.logger.info(f"Time required to process {len(iterator)} entities: {datetime.timedelta(seconds=int(time.time()-total_time))}\n")
    
    config.logger.info("Testing model".center(60, "-"))
    total_time = time.time()

    model.eval()
    out = {}
    with torch.no_grad():
        # Go through each basin
        iterator = tqdm(testing_dataset, desc=f"Testing", unit="basin", ascii=True)
        for basin in iterator:
            loader = DataLoader(
                dataset=testing_dataset[basin],
                batch_size=config.batch_size_evaluation,
                shuffle=False,
                drop_last=False,
                collate_fn=testing_dataset[basin].collate_fn,
                num_workers=config.num_workers
            )

            dates, y_obs, y_hat, params, weights = [], [], [], {}, []
            for sample in loader:
                sample = upload_to_device(sample, config.device)  # upload tensors to device

                dates.append(sample["date"])
                y_obs.append(sample["y_obs"].detach().cpu().numpy())

                # Generate predictions
                batch_params, batch_weights = model(sample).values()
                for k, v in batch_params.items():
                    params[k] = params.get(k, []) + [v.detach().cpu().numpy()]

                weights.append(batch_weights.detach().cpu().numpy())

                pred = model.sample(sample, 1)
                pred[:, :, 0, :] = model.mean(sample)

                # backtransformed information
                pred = pred * testing_dataset[basin].scaler["y_std"].to(config.device) + (testing_dataset[basin].scaler["y_mean"].to(config.device))
                y_hat.append(pred.detach().cpu().numpy())

                # remove from cuda
                del sample, pred
                torch.cuda.empty_cache()

            out[basin] = {
                "date": np.concatenate(dates),
                "y_obs": np.concatenate(y_obs),
                "y_hat": np.concatenate(y_hat),
                "params": {k: np.concatenate(v) for k, v in params.items()},
                "weights": np.concatenate(weights)
            }
            del dates, y_obs, y_hat, params, weights


        config.logger.info(f"Total testing time: {datetime.timedelta(seconds=int(time.time()-total_time))}")

        return to_xarray(out)

def generate_metrics():
    pass

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()

    # Save?
    save_netcdf = args.save

    # Paths
    path_input = Path(args.filepath)    

    match path_input.suffix:

        case ".nc":
            path_experiment = path_input.parent
            config = Config(path_experiment / "config.yml")
            config.logger.info(f" Loading results for: {path_experiment.name} ".center(60, "-"))

            ds = xr.open_dataset(path_input)

        case _:
            path_experiment = path_input.parent.parent

            config = Config(path_experiment / "config.yml")
            config.logger.info(f" Postprocessing model epoch: {path_input.name.split('_')[-1]} ".center(60, "-"))

            ds = generate_netcdf(config, path_input)
            if save_netcdf:
                compression = {var: {"zlib":True, "complevel":5, "dtype":"f4"} for var in ds.data_vars}
                ds.to_netcdf(path_experiment / "results.nc", encoding=compression)

    # We have 'results' and will create or read a .json of metrics in path_experiment.parent
    path_metrics = path_experiment.parent / "metrics.json"
    if path_metrics.exists():
        with open(path_metrics, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    name = path_experiment.name
    if name not in metrics:
        metrics[name] = {}

        basin_ids = [str(basin) for basin in ds.coords["basin_id"].values]

        # Read Scaler
        with open(path_experiment / "scaler.pickle", "rb") as f:
            scaler = pickle.load(f)
        y_mean, y_std = scaler["y_mean"].numpy(), scaler["y_std"].numpy()

        for basin in tqdm(basin_ids, ascii=True):
            # Read basin data
            data = ds.sel(basin_id=basin)

            y_hat_mean = data.y_hat[:, -1, 0, 0]

            # Metrics
            metrics[name][basin] = {}
            metrics[name][basin]["NSE"] = calc_nse(data.y_obs[:, -1, 0], y_hat_mean)
            metrics[name][basin]["LOGLIK"] = {"True": float("nan")}
            with torch.no_grad():
                dist = Distribution.from_string(config.distribution)
                params = {"loc": data["loc"].values, "scale": data["scale"].values}
                if dist == Distribution.LAPLACIAN:
                    params["kappa"] = data["kappa"].values
                params = {k: torch.tensor(v) for k, v in params.items()}
                weights = torch.tensor(data["weights"].values)
                y_obs = torch.tensor((data["y_obs"].values - y_mean) / y_std)

                loglik = -1 * loss_nll(params, weights, dist, y_obs)
                metrics[name][basin]["LOGLIK"]["True"] = loglik.item()

    with Path.open(path_metrics, "w") as f:
        json.dump(metrics, f, indent=4)