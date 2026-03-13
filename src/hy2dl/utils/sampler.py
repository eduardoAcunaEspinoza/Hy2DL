import numpy as np
from torch.utils.data import Sampler


class GaugeBatchSampler(Sampler):
    """Produces batches of indices that belong to the same gauge_id

    This sampler ensures that all samples in a batch come from the same gauge_id. This is necessary in evaluation
    (validation/testing) where we need to process the data for each gauge separately.

    Parameters
    ----------
    valid_samples : numpy.ndarray
        A 1D structured numpy array containing the valid samples.
        Fields are:
        - 'gauge_id' (object): The ID of the basin/gauge.
        - 'date' (datetime64[ns]): The timestamp of the sample.
        - 'source' (object): The data source (e.g., 'obs' or 'fc').
    batch_size : int
        Size of each batch

    Note: valid_samples must be sorted by 'gauge_id' and 'date': np.sort(self.valid_samples, order=["gauge_id", "date"])

    """

    def __init__(self, valid_samples: np.ndarray, batch_size: int):
        self.batch_size = batch_size

        # Computes the boundaries for the different gauge_ids. Valid samples were previously sorted by gauge_id and date
        unique_gauges, start_indices = np.unique(valid_samples["gauge_id"], return_index=True)
        boundaries = np.append(start_indices, len(valid_samples))

        self.batches = []
        # Construct batches
        for i in range(len(unique_gauges)):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Create a sequence of indices for this specific basin
            basin_indices = np.arange(start_idx, end_idx)

            # Split into chunks and append blindly
            for j in range(0, len(basin_indices), self.batch_size):
                batch = basin_indices[j : j + self.batch_size].tolist()
                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
