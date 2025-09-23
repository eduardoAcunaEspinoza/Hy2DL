Modelzoo
========

The following section gives an overview of all implemented models. 


Model Classes
-------------

CudaLSTM
^^^^^^^^
:py:class:`hy2dl.modelzoo.cudalstm.CudaLSTM` uses the standard PyTorch LSTM implementation. 

An example using this model can be found in the notebook folder, in the github repository.

Hybrid-Model
^^^^^^^^^^^^
:py:class:`hy2dl.modelzoo.hybridmodel.HybridModel` is a wrapper class to combine data-driven methods with
conceptual hydrological models. Specifically, an LSTM network is used to produce a dynamic parameterization for a
conceptual hydrological model. One can use multiple entities of the conceptual hydrological model acting in parallel.

Currently available conceptual hydrological models are:

-   :py:class:`hy2dl.modelzoo.shm.SHM`: `Acuna Espinoza et al (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_
-   :py:class:`hy2dl.modelzoo.linear_reservoir.linear_reservoir`: `Acuna Espinoza et al (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_
-   :py:class:`hy2dl.modelzoo.nonsense.NonSense`: `Acuna Espinoza et al (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_
-   :py:class:`hy2dl.modelzoo.hbv.HBV`: `Feng et al (2022) <https://doi.org/10.1029/2022WR032404>`_ , `Acuna Espinoza et al (2025) <https://doi.org/10.5194/hess-29-1277-2025>`_


There is also the option to include an extra routing method (after the conceptual hydrological model), using a unit hydrograph based on gamma function
:py:class:`hy2dl.modelzoo.uh_routing.UH_routing`. 

For more information about hybrid models we refer to `Acuña Espinoza et al (2024) <https://doi.org/10.5194/hess-28-2705-2024>`__. An example using this model can be found in the notebook folder, 
in the github repository.


MF-LSTM: Multi-frequency LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By combining  :py:class:`hy2dl.modelzoo.cudalstm.CudaLSTM` and :py:class:`hy2dl.modelzoo.inputlayer.InputLayer`, one can use a single-cell
lstm to process data at multiple temporal resolutions (e.g. hourly and daily). 

For reference see: `Acuna Espinoza et al (2025b) <https://doi.org/10.5194/hess-29-1749-2025>`_

An example using this model can be found in the notebook folder, in the github repository.


Forecast-LSTM: Forecast LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`hy2dl.modelzoo.forecast_lstm.ForecastLSTM` a single LSTM cell rolls out through the hindcast and forecast period. Different embedding layers are used in each period
to handle different amount of variables or varying types and quality of data. The model supports different temporal frequencies in the hindcast period.

An example using this model can be found in the notebook folder, in the github repository.

LSTM-MDN: LSTM with Mixture Density Network output layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`hy2dl.modelzoo.lstmmdn.LSTM-MDN` combines an LSTM network with a Mixture Density Network (MDN) output layer. The MDN layer allows to model the output as a mixture of probability distributions, enabling probabilistic predictions.

For reference see: `Klotz et al (2022) <https://doi.org/10.5194/hess-26-1673-2022>`_

An example using this model can be found in the notebook folder, in the github repository.
