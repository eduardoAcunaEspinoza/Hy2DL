Configuration Arguments
=======================

To define an experiment configuration, you can provide either a .yml file or a Python dictionary.
This page lists all available configuration arguments and their descriptions.

For practical reference, example configurations are available in the examples folder of the GitHub repository.

Experiment configuration
---------------------------------

- ``device`` (str): 
    Name of the device to be used for training. Can be either "cpu", "gpu" or "cuda:#", where # is the GPU id.

- ``experiment_name`` (str): 
    Name of the experiment.

- ``path_save_folder`` (str): 
    Path to the directory where experiment results will be stored.  If not specified, a folder will automatically be created at: ``../results/<experiment_name>_seed_<random_seed>``

- ``random_seed`` (int): 
    Random seed for reproducibility. If not specified, a random seed will be generated automatically

- ``testing_period`` (list[str]): 

    Initial and final date of the testing period. It should be consistent with the resolution of the data.

     - Example daily resolution: ["1990-10-01", "2003-09-30"] 
     - Example daily resolution: ["1990-10-01 00:00:00", "2003-09-30 23:00:00"] 

- ``training_period`` (list[str]): 
    Initial and final date of the training period. It should be consistent with the resolution of the data.

- ``validation_period`` (list[str]): 
    Initial and final date of the validation period. It should be consistent with the resolution of the data.

Data settings
-----------------------------

- ``dataset`` (str): 
    Name of the dataset to be used. Current options are: "camels_us", "camels_gb", "camels_de", "caravan", "hourly_camels_us", "hourly_camels_de".

- ``dynamic_input`` (list[str] | dict[str, list[str] | dict[str, list[str]]]): 
    
    Name of variables used as dynamic series input in the data driven model. In most cases it is a single list.

     - Case 1: A single list of variable names.
     - Case 2 (multiple frequencies): Use a dictionary where each key is a frequency id and each value is a list of variables.
     - Case 3 (single frequency with groups): use a dictionary where each key is a group id and each value is a list of variables.
     - Case 4 (multiple frequencies with groups): use a nested dictionary where each frequency maps to its groups, and each group maps to a list of variables.

.. code-block:: python

    # First case: using a single list
    dynamic_input = ["total_precipitation", "temperature"]

    # Second case: different inputs for each frequency
    dynamic_input = {
        "1D": ["prcp(mm/day)_daymet", "tmax(C)_daymet", "tmin(C)_daymet"],
        "1h": ["total_precipitation_nldas_hourly", "temperature_nldas_hourly"]
    }

    # Third case: different groups of variables
    dynamic_input = {
        "g1": ["prcp(mm/day)_daymet", "tmax(C)_daymet", "tmin(C)_daymet"],
        "g2": ["prcp(mm/day)_maurer_extended", "tmax(C)_maurer_extended", "tmin(C)_maurer_extended"]
    }

    # Fourth case: different groups for each frequency
    dynamic_input = {
        "1D": {
            "g1": ["prcp(mm/day)_daymet", "tmax(C)_daymet", "tmin(C)_daymet"],
            "g2": ["prcp(mm/day)_maurer_extended", "tmax(C)_maurer_extended", "tmin(C)_maurer_extended"]
        },
        "1h": {
            "g3": ["total_precipitation_nldas_hourly", "temperature_nldas_hourly"],
            "g2": ["prcp(mm/day)_maurer_extended", "tmax(C)_maurer_extended", "tmin(C)_maurer_extended"]
        }
    }

- ``lagged_features`` (dict[str, int | list[int]]): 
    Allows adding lagged copies of existing variables. Specify as a dictionary where the **key** is the variable name and the **value** is the lag. 
    The lag can be a single integer for a single lag or a list of integers for multiple lags. The resulting lagged variable(s) will be named ``<variable_name>_shift<N>``, 
    where ``<variable_name>`` is the original variable name and ``N`` is the lag value. Any lagged variable must also be included in the ``dynamic_input`` argument, with the
    ``<variable_name>_shift<N>`` format.  

- ``path_data`` (str):
    Path to the directory containing the input data files.

- ``path_additional_features`` (str):
    Path to a dictionary, where each key is an entity id and each value is a date-indexed dataframe containing additional features as columns. This is a way
    to add additional variables (e.g. evapotranspitation) that are not included in the main dataset files.

- ``path_entities`` (str): 
    Path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed. If one wants to use different
    entities for training, validation and testing, one can use the keywords ``path_entities_training``, ``path_entities_validation`` and ``path_entities_testing``.

- ``static_input`` (list[str]): 
    Name of static attributes used as input in the model (e.g. catchment attributes).

- ``target`` (list[str]):
    Target variable that will be used to train the model

Training settings
-----------------------------

- ``batch_size_training`` (int): 
    Batch size used during training.

- ``batch_size_evaluation`` (int): 
    Batch size used during evaluation (validation and testing). If not specified, ``batch_size_training`` will be used.

- ``dropout_rate`` (float): 
    Dropout rate used in the model.

- ``epochs`` (int): 
    Number of epochs used during training.

- ``learning_rate`` (int): 
    Learning rate used during training. It can be a single float value. It can  be a dictionary specifying in which epochs the learning rate will change. 
    It can also be a single float value, but with an StepLR scheduler defined by ``steplr_step_size`` and ``steplr_gamma``.

.. code-block:: python

    # First case: single float value
    learning_rate = 0.001

    # Second case: dictionary with epoch-specific values
    learning_rate = {
        1: 0.001,
        10: 0.0005,
        20: 0.0001
    }

    # Third case: single float with scheduler
    learning_rate = 0.001
    steplr_step_size = 5
    steplr_gamma = 0.8

- ``max_updates_per_epoch`` (int): 
    Maximum number of updates per epoch. Useful if one does not want to use all the training data in each epoch.

- ``num_workers`` (int):
    Number of (parallel) threads used in the data loader. Default is 0. Use 0 when debugging the code.

- ``predict_last_n`` (int):
    Number of timesteps of the sequence length are used for prediction. Default is 1. 

- ``optimizer`` (str):
    Name of the optimizer to use during training. Currently only ADAM is implemented.

- ``seq_length`` (int):
    Length of the input sequence.

- ``unique_prediction_blocks`` (bool):
    If True, the training data is divided into unique prediction blocks (no overlap between training blocks). Default is False.

Evaluation settings
-----------------------------

- ``validate_every`` (str):
    Number of epochs after which the model will be validated. Default is 1.

- ``validate_n_random_basins`` (str):
    Number of random basins used for validation. If 0, all basins will be used. Default is 0.


Model configuration
-----------------------------

- ``model`` (str):
    Name of the model to use. Currently implemented: "cudalstm", "forecast_lstm", "hybrid", "lstmmdn".

- ``hidden_size`` (int):
    Number of hidden units in the LSTM cell

- ``initial_forget_gate`` (float):
    Initial value of the forget gate bias.


Embedding networks
-----------------------------
- ``dynamic_embedding`` (dict[str, str | float | list[int]]): 

    Configuration for dynamic embedding layers. Specify as a dictionary with the keys:

     - hiddens: List of integers defining number of neurons per layer.
     - activation: Activation function to be use. The same activation function will be used for all layers, except the last one, where no activation is used. Currently supported are: relu, tanh, sigmoid and linear. Default relu.
     - dropout: Dropout rate applied after each layer, except the last one. Default: 0.0

.. code-block:: python

    # Example dynamic embedding
    dynamic_embedding = {
        "hiddens": [10, 10, 10],
        "activation": "relu",
        "dropout": 0.0
    }

- ``nan_handling_method`` (str):
    Method to handle NaN values in the input data. To use this argument, groups of variables need to be defined in ``dynamic_input``. 
    Currently supported are: masked_mean and input_replacement. For details, see: `Gauch et al (2025) <https://doi.org/10.5194/egusphere-2025-1224>`_

- ``nan_probability`` (dict[str, dict[str, float]]):
    Method to included NaN values during training to make the model more robust to missing values during inference. To use this argument, groups of variables need to be
    defined in ``dynamic_input``. The ``nan_probability`` argument should be specify as a nested dictionary where the first key is the group id defined in ``dynamic_input``,
    and each value is another dictionary with the keys ``nan_seq`` and ``nan_step`` to indicate the probabilty of masking an entire sequence or a single timestep, respectively.
    For details, see: `Gauch et al (2025) <https://doi.org/10.5194/egusphere-2025-1224>`_

.. code-block:: python

    # Example nan_probability
    nan_probability = {
        "g1": {"nan_seq": 0.0, "nan_step": 0.0},
        "g2": {"nan_seq": 0.1, "nan_step": 0.1}
    }

- ``nan_probabilistic_masking`` (bool):   
    Boolean to specify is probabilistic masking should be used. In this case ``nan_handling_method`` and ``nan_probability`` must be specified. Useful to turn on/off masking
    for training/evaliuation.

- ``output_features`` (int):
    Number of output features for linear layer after LSTM. Default is 1 (e.g. deterministic discharge prediction)

- ``static_embedding`` (dict[str, str | float | list[int]]): 
    Configuration for static embedding layers (embedding for static attributes). It has the same structure as ``dynamic_embedding``.


Hybrid model
-----------------------------
- ``conceptual_model`` (str):
    Name of the hydrological conceptual model that is used together with a data-driven method to create the hybrid model. 
    Currently implemented: "shm", "linear_reservoir", "nonsense", "hbv".

- ``dynamic_input_conceptual_model`` (dict[str, list[str]]):
    Dictionary mappping the variables taken as input by the conceptual model to variables in our dataset. If one specify multiple dataset variables, the 
    average of this variables will be taken as input for the conceptual model.

.. code-block:: python

    # Example dynamic input for conceptual model
    dynamic_input_conceptual_model = {
        "precipitation": "prcp(mm/day)",
        "pet": "pet(mm/day)",
        "temperature": ["tmax(C)", "tmin(C)"]
    }

- ``dynamic_parameterization_conceptual_model`` (list[str]):
    List to specify which of the parameters of the conceptual model will be dynamic. That is, which parameters will vary in time. 
    If not specifiy, the parameter is taken as static.

- ``num_conceptual_models`` (int):
    Number of conceptual models that will run on parallel. The LSTM  will provide static or dynamic parametrization for each of the "n" conceptual models 
    and the output of these models is combined to get the final discharge prediction. The default is 1.

- ``dynamic_parameterization_conceptual_model`` (list[str]):
    List to specify which of the parameters of the conceptual model will be dynamic. That is, which parameters will vary in time. 
    If not specifiy, the parameter is taken as static.

- ``routing_model`` (str):
    Name of the additional routing model that will be used after the conceptual model. Currently only "uh_routing" is available, where a unit hydrograph based on the gamma function is used.


Multi-frequency LSTM (MF-LSTM)
--------------------------------

- ``custom_seq_processing`` (dict[str, dict[str, int]]):
    
    Dictionary specifying how to process the input sequences for each frequency. The keys are the frequency names (e.g., "1D", "1h"), and the values are dictionaries with the following keys:
    
    - n_steps (int): Number of timesteps to process in this frequency.
    - freq_factor (int): Factor to convert to the respective frequencies (using the highest frequency as base).

    The sum of n_steps*freq_factor for all the frequencies should match the specified ``seq_length``. In case different variables are used of each frequency, the keys in ``custom_seq_processing`` 
    should match the keys in ``dynamic_input``.

.. code-block:: python

    # I use as sequence length one year of hourly data (365*24)
    seq_length = 8760

    # I process 360 days at daily resolution (24 h blocks) and 96 hours (4 days) days at hourly resolution.
    custom_seq_processing = {
        "1D": {"n_steps": 360, "freq_factor": 24},
        "1H": {"n_steps":96, "freq_factor": 1}
    }

- ``custom_seq_processing_flag`` (bool):
    Boolean to specify if binary flags should be used to indicate the different frequencies. Default is False.

Forecast model
-----------------------------
- ``forecast_input`` (list[str] | dict[str, list[str]]:):
    Input list or dictionary specifying the variables used in forecast section. Currently forecast input is only supported for single frequency, and the dictionary
    case is used if multiple groups of variables are present. See first and third case from the example of ``dynamic_input``.

- ``seq_length_hindcast`` (int):
    Length of the input sequence for hindcast period.

- ``seq_length_forecast`` (int):
    Length of the input sequence for forecast period.


Probabilistic model
-----------------------
Documentation in progress.


CAMELS US specific
-----------------------
- ``forcings`` (list[str]):
    Name of forcing product from CAMELS US dataset that will be included. Available products are: "daymet", "ndlas", "maurer",
    "ndlas_extended", "maurer_extended" and for "hourly_camels_us" also "ndlas_hourly" is available.