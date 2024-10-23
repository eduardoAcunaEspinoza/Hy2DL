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

-   :py:class:`hy2dl.modelzoo.hbv.HBV`
-   :py:class:`hy2dl.modelzoo.shm.SHM`
-   :py:class:`hy2dl.modelzoo.linear_reservoir.linear_reservoir`
-   :py:class:`hy2dl.modelzoo.nonsense.NonSense`

There is also the option to include an extra routing method (after the conceptual hydrological model), using a unit hydrograph based on gamma function
:py:class:`hy2dl.modelzoo.uh_routing.UH_routing`. For more information about hybrid models we refer to `Acuña Espinoza et al (2024) <https://doi.org/10.5194/hess-28-2705-2024>`__.

An example using this model can be found in the notebook folder, in the github repository.


MF-LSTM: Multiple-Frequency LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`hy2dl.modelzoo.mflstm.MFLSTM` uses a single-cell lstm to process data at multiple temporal resolutions (e.g. hourly and daily).

An example using this model can be found in the notebook folder, in the github repository.