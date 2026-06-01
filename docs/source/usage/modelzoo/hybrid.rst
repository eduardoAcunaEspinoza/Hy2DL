Hybrid models
=============

The :py:class:`hy2dl.modelzoo.hybrid.Hybrid` wrapper class combines data-driven methods with conceptual hydrological models. 

An LSTM network is used to parameterize the conceptual hydrological model. Details of the configuration arguments related to this model can be found in :ref:`hybrid_reference`

The following illustrative example demonstrates how the model functions: Imagine you are interested in simulating daily streamflow for the year 1997. The hybrid model operates using a sequence-to-sequence (seq2seq) approach. This means the LSTM generates a parameter set for each time step, which then feeds into the conceptual hydrological model.

   .. figure:: ../../_static/hybrid_model.png
      :alt: Sequence-to-sequence hybrid model diagram
      :align: center

Due to the nature of conceptual models, a warm-up period is necessary to initialize the model states. As shown in the figure below, this warm-up period is defined as the difference between ``seq_length`` and ``predict_last_n``. This period is solely used to initialize the states of the conceptual model and does not contribute to the loss function during training.

   .. figure:: ../../_static/hybrid_warmup_seq.png
      :alt: Diagram illustrating the warm-up sequence
      :align: center


Conceptual models
-----------------
The currently available conceptual hydrological models are:

-   :py:class:`hy2dl.modelzoo.shm.SHM`: `Acuña Espinoza et al. (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_ , `Alvarez Chaves et al. (2026) <https://doi.org/10.5194/hess-30-629-2026>`_
-   :py:class:`hy2dl.modelzoo.linear_reservoir.linear_reservoir`: `Acuña Espinoza et al. (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_, `Alvarez Chaves et al. (2026) <https://doi.org/10.5194/hess-30-629-2026>`_
-   :py:class:`hy2dl.modelzoo.nonsense.NonSense`: `Acuña Espinoza et al. (2024) <https://doi.org/10.5194/hess-28-2705-2024>`_, `Alvarez Chaves et al. (2026) <https://doi.org/10.5194/hess-30-629-2026>`_
-   :py:class:`hy2dl.modelzoo.hbv.HBV`: `Feng et al. (2022) <https://doi.org/10.1029/2022WR032404>`_, `Acuña Espinoza et al. (2025) <https://doi.org/10.5194/hess-29-1277-2025>`_

You also have the option to include an additional routing method (applied after the conceptual model) using a unit hydrograph based on a gamma function via :py:class:`hy2dl.modelzoo.uh_routing.UH_routing`. 


Dynamic vs. static parameterization
-----------------------------------
Using the ``dynamic_parameterization_conceptual_model`` parameter, you can define which conceptual model parameters (defined in the ``@property`` ``parameter_ranges`` of each model) are dynamic (varying at each time step) or static (remaining constant over time). 

The figure below illustrates both cases. Because the LSTM naturally outputs one parameter set per time step, we approximate a static value by calculating the mean of the parameter and repeating it across all time steps.
   
   .. figure:: ../../_static/hybrid_dynamic_vs_static.png
      :alt: Comparison of dynamic and static parameterizations
      :align: center

**Note:** Even when using dynamic parameterization, the warm-up period always utilizes a static value.