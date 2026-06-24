"""Tester for self-supervised LSTMs with configurable input-masking scenarios."""

import time

import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from hy2dl.datasetzoo import BaseDataset
from hy2dl.evaluation.simulation_tester import SimulationTester
from hy2dl.utils.config import Config
from hy2dl.utils.utils import upload_to_device


class SSLTester(SimulationTester):
    """Tester that evaluates the model on one or more input-masking scenarios.

    For each entry in ``cfg.eval_scenarios`` the standard simulation evaluation
    loop is run with the listed variables forced to NaN at the input. The model
    replaces NaN with its learned mask token, so this effectively asks the
    model to reconstruct the masked variables from the unmasked ones plus
    static attributes.

    Each scenario writes its predictions to a separate zarr file named
    ``<period>_results_<scenario_name>.zarr`` under ``cfg.path_save_folder``.

    Configuration example
    ---------------------
    ::

        eval_scenarios:
          - name: single_NO3
            mask_vars: [NO3.N..mg.l._WQ]
          - name: all_targets
            mask_vars: [discharge_spec_obs, NO3.N..mg.l._WQ, PO4.P..mg.l._WQ,
                        O2..mg.l._WQ, LF..mueS.cm._WQ, WT..C._WQ,
                        DOC..mg.l._WQ]
          - name: no_masking
            mask_vars: []

    If ``cfg.eval_scenarios`` is not set, a single scenario named ``"default"``
    with an empty ``mask_vars`` list is used (no forced masking).

    Notes
    -----
    * Per-epoch validation is disabled for this tester (``validate_model`` is a
      no-op). Validation across multiple masking scenarios is expensive, and
      the SSL evaluation metrics only become meaningful at the end of training.
    * Variables in ``mask_vars`` that are not present in ``sample["x_d"]`` are
      skipped, with a one-shot warning per scenario logged via
      ``cfg.logger.warning``. This lets the same scenario list be reused across
      configs with slightly different ``dynamic_input`` while still surfacing
      typos.

    Parameters
    ----------
    cfg : Config
        Configuration object containing model hyperparameters and settings.
    evaluation_dataset : BaseDataset
        Dataset used for evaluation.
    """

    def __init__(self, cfg: Config, evaluation_dataset: BaseDataset):
        super().__init__(cfg=cfg, evaluation_dataset=evaluation_dataset)
        self._base_path_zarr = self.path_zarr

    def _scenarios(self) -> list[dict]:
        scenarios = self.cfg.eval_scenarios
        if not scenarios:
            return [{"name": "default", "mask_vars": []}]
        return scenarios

    def _scenario_zarr_path(self, scenario_name: str):
        return self._base_path_zarr.with_name(f"{self._base_path_zarr.stem}_{scenario_name}.zarr")

    def evaluate_model(self, model: torch.nn.Module):
        """Evaluate the model on every configured scenario and write per-scenario zarr files."""
        for scenario in self._scenarios():
            self.path_zarr = self._scenario_zarr_path(scenario["name"])
            self._evaluate_scenario(model=model, mask_vars=scenario.get("mask_vars", []) or [])
        self.path_zarr = self._base_path_zarr

    def _evaluate_scenario(self, model: torch.nn.Module, mask_vars: list[str]):
        """Run the standard evaluation loop with NaN injection for the configured mask_vars."""
        model.eval()
        self._initialize_zarr()
        self.gauge_data = {k: [] for k in self.gauge_data}
        current_gauge = None

        # Warn once per scenario for any mask_var that is missing from sample["x_d"], so
        # silent no-ops (e.g. due to typos in the scenario config) are at least visible.
        warned_missing: set[str] = set()

        with torch.no_grad():
            iterator = tqdm(
                self.evaluation_loader,
                desc=f"Eval [{self.path_zarr.stem}]",
                unit="batches",
                ascii=True,
                leave=False,
            )

            with threadpool_limits(limits=1):
                for sample in iterator:
                    sample = upload_to_device(sample, self.cfg.device)
                    batch_gauge_id = sample["gauge_id"][0]

                    if current_gauge is not None and batch_gauge_id != current_gauge:
                        self._write_to_zarr(current_gauge)
                        self.gauge_data = {k: [] for k in self.gauge_data}

                    current_gauge = batch_gauge_id

                    if mask_vars:
                        x_d_modified = {var: t.clone() for var, t in sample["x_d"].items()}
                        for var in mask_vars:
                            if var in x_d_modified:
                                x_d_modified[var] = torch.full_like(x_d_modified[var], float("nan"))
                            elif var not in warned_missing:
                                self.cfg.logger.warning(
                                    f"[{self.path_zarr.stem}] mask_var {var!r} is not in "
                                    f"sample['x_d'] (dynamic_input) — skipping. Check for typos "
                                    f"in eval_scenarios."
                                )
                                warned_missing.add(var)
                        sample_for_model = {**sample, "x_d": x_d_modified}
                    else:
                        sample_for_model = sample

                    pred = model(sample_for_model)
                    self._process_results(sample, pred)
                    del sample, sample_for_model, pred

            if current_gauge is not None:
                self._write_to_zarr(current_gauge)

        zarr.consolidate_metadata(self.path_zarr)

    def validate_model(self, *_args, **_kwargs) -> None:
        """Per-epoch validation is disabled for SSL — see class docstring."""
        # Keep validation_report empty so the trainer's logging template still works
        empty_metrics = "".join([f"{'':^10}|" for _ in range(len(self.cfg.validation_metric))])
        self.validation_report = empty_metrics + f"{'':^10}|"
