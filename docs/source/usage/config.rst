Configuration Arguments
=======================

This page provides a list of possible configuration arguments. Examples involving different configurations 
can in the notebook folder, in the github repository.

General experiment configuration
---------------------------------

-   ``experiment_name``: Name of the experiment

-   ``path_entities``: Path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed. An 
    alternative option is to specify the entity directly with the ``entity`` parameter.

-   ``dynamic_input``: Name of variables used as dynamic series input in the data driven model. In most cases is a single list. If we 
    are using multiple frequencies, and each frequency have different number of inputs, it is a dictionary where the key is the 
    frequency and the value is a list of variables.

.. code-block::

    # First case, using a single list
    dynamic_input = ["total_precipitation", "temperature"]
    # Second case, using different inputs for each frequency
    dynamic_input = {"1D": ["prcp(mm/day)_daymet","tmin(C)_maurer_extended"], "1h": ["convective_fraction_nldas_hourly","longwave_radiation_nldas_hourly","potential_energy_nldas_hourly",]}

-   ``forcing``: Exclusive of CAMELS-US! Specificy which forcing data will be used (e.g. daymet, maurer, ndlas, ndlas_hourly, etc.)

-   ``target``: Target variable that will be used to train the model

-   ``static_input``: Name of static attributes used as input in the lstm (e.g. catchment attributes).

-   ``training_period``: Initial and final date of the training period of interest. It should be consistent with the resolution of the data.
    e.g.  daily: ["1990-10-01","2003-09-30"], hourly: ["1990-10-01 00:00:00","2003-09-30 23:00:00"]

-   ``validation_period``: Initial and final date of the validation eriod of interest. It should be consistent with the resolution of the data.
    e.g.  daily: ["1990-10-01","2003-09-30"], hourly: ["1990-10-01 00:00:00","2003-09-30 23:00:00"]

-   ``testing_period``: Initial and final date of the testing eriod of interest. It should be consistent with the resolution of the data.
    e.g.  daily: ["1990-10-01","2003-09-30"], hourly: ["1990-10-01 00:00:00","2003-09-30 23:00:00"]

-   ``running_device``: "cpu" or "gpu", depending on where the model will be run.

-   ``seed``: Integer that will be used to set the seed. Useful to reproduce the results.

General model configuration
-----------------------------

-   ``model_configuration[batch_size_training]``: Batch size used during training

-   ``model_configuration[batch_size_evaluation]``: Batch size used during evaluation (validation and testing)

-   ``model_configuration[no_of_epochs]``: Number of epochs used during training

- ``model_configuration[learning_rate]``: Learning rate used during training. It can be a single float value. It can also be a dictionary specifying in which epochs the learning rate will change. As a third option, it can be a single float value, but with an StepLR scheduler. 

.. code-block::

    # First case, single float value
    model_configuration[learning_rate] = 0.001
    # Second case, dictionary
    model_configuration[learning_rate] = {1: 1e-3, 10: 5e-4, 20: 1e-4}
    # Third case, single float value with StepLR scheduler. One should include as additional arguments
    # adapt_learning_rate_epoch and adapt_gamma_learning_rate.
    model_configuration{learning_rate: 0.001, adapt_learning_rate_epoch: 10, adapt_gamma_learning_rate = 0.1}

-  ``model_configuration[hidden_size]``: Number of hidden units in the LSTM cell

-  ``model_configuration[seq_length]``: Length of input sequence

-  ``model_configuration[predict_last_n]``: Which timesteps of the sequence length are used for prediction. Implementing seq-to-one would mean predict_last_n = 1. Implementing seq-to-seq would mean predict_last_n > 1. It always need to be smaller to the sequence length.

-  ``model_configuration[dropout_rate]``: Dropout rate used in the LSTM cell

-  ``model_configuration[validate_every]``: Number of epochs after which the model will be validated

-  ``model_configuration[validate_n_random_basins]``: Number of random basins used for evaluation. If -1, all basins will be used.

-  ``model_configuration[max_updates_per_epoch]``: Maximum number of updates per epoch. Useful if one does not want to use all the training data in each epoch.

Configuration CudaLSTM
-----------------------------

-  ``model_configuration[n_dynamic_channels_lstm]``: Number of channels in the lstm to handle the dynamic input. Usually defined as len(dynamic_input). 


Configuration Hybrid models
-----------------------------

-  ``conceptual_input``: List of features that are used as input in the conceptual part of the hybrid model. Given how the conceptual models are implemented, the first entity of the list must be precipitation and the 
   second entity potential evapotranspitation. For temperature there are two options. The first option is to specify as the third entity of the list the mean/average temperature. The second option is to specify as the 
   third and forth entities of the list the minimum and maximum temperature, which will later be averaged by the conceptual models to  estimate the mean temperature.

- ``model_configuration[input_size_lstm]``: Input size of the LSTM model. For hybrid models it is defined as len(dynamic_input) + len(static_input). 

- ``model_configuration[conceptual_model]``: Name of the hydrological conceptual model that is used together with a data-driven method to create the hybrid model e.g., [``SHM``, ``HBV``].

- ``model_configuration[n_conceptual_models]``: Number of conceptual models that will run on parallel. The LSTM  will provide static or dynamic parametrization for each of the "n" conceptual models and the output of these models is combined to get the final discharge prediction. The default is 1.

-  ``model_configuration[conceptual_model_dynamic_parameterization]``: List to specify which of the parameters of the conceptual model will 
   be dynamic. That is, which parameters will vary in time. If not specifiy, the parameter is taken as static.

- ``model_configuration[routing_model]``: Name of the additional routing model that will be used after the conceptual model. Currently only :py:class:`hy2dl.modelzoo.uh_routing.UH_routing` is available.

- ``model_configuration[unique_prediction_blocks]``: True if one wants to divide the training data in unique_prediction_blocks (no overlap between training blocks).
    

Configuration MF-LSTM
-----------------------

-  ``model_configuration[n_dynamic_channels_lstm]``: Number of channels in the lstm to handle the dynamic input. Can be len(dynamic_input) if all frequencies have the same number of dynamic inputs. Otherwise one should specify the value.

-  ``model_configuration[custom_freq_processing]``: Dictionary specifying the keys and values for the number of steps and the frequency factor, for each frequency. The number of steps is the number of timesteps that will processed in each frequency, and the frequency factor 
   indicates how to convert to the respective frequencies (using the highest frequency as base).

.. code-block::

    # First case, single float value
    model_configuration[custom_freq_processing] = {"1D": {"n_steps": 351,"freq_factor": 24,},"1h": {"n_steps": (365 - 351) * 24, "freq_factor": 1}}

- ``model_configuration[unique_prediction_blocks]``: True if one wants to divide the training data and evaluation in unique_prediction_blocks (no overlap between blocks).

- ``model_configuration[dynamic_embeddings]``: True in case one wants to use a fully connected networks as an embedding layer for the dynamic input. Necessary when the number of dynamic inputs for each frequency is different.

