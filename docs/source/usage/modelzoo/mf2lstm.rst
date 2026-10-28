MF2LSTM
=======

:py:class:`hy2dl.modelzoo.mf2lstm.MF2LSTM` implements the Multi-Frequency Masked-Forecasting LSTM architecture. This model integrates multi-frequency (:ref:`multi_frequency_reference`) hindcast processing with a sequential-forecasting (:ref:`forecast_reference`) framework for computationally efficient hydrological forecasting. It relies on a masked-mean (:ref:`masked_mean_reference`) embedding strategy to handle missing data and assimilate real-time discharge signals.

The model processes data across two distinct phases: a hindcast period (historical observations) and a forecast period. Multi-frequency data-handling is only supported on the hindcast period. Depending on whether the model is in training or evaluation mode, it processes the forecast sequence differently:

- Training Mode (Teacher-Forcing): Uses observed discharge targets across the forecast horizon to accelerate training. A sigmoid-scaled synthetic noise profile is injected over the sequence to reduce exposure bias.

   .. figure:: ../../_static/mf2lstm_train.png
      :alt: MF2LSTM architecture overview
      :align: center

      MF2LSTM using teacher-forcing scheme

- Inference Mode (Autoregressive): Manually unrolls the single LSTM cell across the forecast horizon, recursively feeding predicted discharges back into subsequent timesteps.

   .. figure:: ../../_static/mf2lstm_eval.png
      :alt: MF2LSTM architecture overview
      :align: center

      MF2LSTM in autoregressive mode

Data assimilation
------------------
   We repurpose the `masked-mean <https://doi.org/10.5194/hess-29-6221-2025>`_ embedding strategy for data assimilation. We define multiple groups, one of which contains the autoregressive information (e.g., target information), which are then embedded and masked (:ref:`masked_mean_reference`). During training, step and sequence masking (:ref:`nan_reference`) of the autoregressive input teaches the network how to handle missing values, enabling robust real-time data assimilation as well as seamless fallback operation in ungauged scenarios.

   .. figure:: ../../_static/mf2lstm_stepnan.png
      :alt: Step masking
      :align: center

      Step masking

   .. figure:: ../../_static/mf2lstm_seqnan.png
      :alt: Sequence masking
      :align: center

      Sequence masking

Additional
----------
   - Optional step counters (``forecast_counter``) can be concatenated along the feature dimension to provide explicit positional encoding across the hindcast and forecast horizons (supporting linear step counting or exponential decay via ``forecast_counter_tau``).
   - This model supports :ref:`mdn_reference` for probabilistic output.