Getting started
================

The package can be installed using ``uv`` or ``pip``. We recommend the former as it provides faster installation times and better dependency management.

To install ``uv``, follow the instructions at `uv's documentation <https://docs.astral.sh/uv/getting-started/installation/>`_.

Installation
------------

You can install the package either using **uv** (recommended) or **pip**. Editable mode is available for development purposes.

Installation with uv
^^^^^^^^^^^^^^^^^^^^^

For a regular installation:

.. code-block:: bash

   uv add hy2dl

For an editable/development installation:

1. Clone the repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/eduardoAcunaEspinoza/Hy2DL.git
      cd Hy2DL

   Alternatively, download the `zip file <https://github.com/eduardoAcunaEspinoza/Hy2DL>`_ from GitHub and unzip it, then navigate to the repository root.

2. Sync the environment and install the package:

   .. code-block:: bash

      uv sync

This will install Hy2DL and all dependencies specified in the `pyproject.toml` file in editable mode.

Installation with pip
^^^^^^^^^^^^^^^^^^^^^^

Note: If you decide to use ``pip``, make sure you already have python ">=3.10" installed in your system.

For a regular installation:

.. code-block:: bash

   pip install hy2dl

For an editable/development installation:

1. Clone the repository from GitHub (or download the zip file as described above):

   .. code-block:: bash

      git clone https://github.com/eduardoAcunaEspinoza/Hy2DL.git
      cd Hy2DL

2. Install the package in editable mode:

   .. code-block:: bash

      pip install -e .

Running experiments
---------------------

After installing the package, you can explore and run different experiments.

In the `GitHub repository <https://github.com/eduardoAcunaEspinoza/Hy2DL>`_, you will find the folder ``notebooks``, which contains several examples that can serve as a starting point. 

Experiment configurations can be specified in two ways:

- As ``.yml`` files
- Directly as a python dictionary

The folder ``examples`` provides ready-to-use configurations that can be directly loaded in the notebooks. In addition, the notebook ``LSTM_Forecast.ipynb`` demonstrates how to define the configuration as a Python dictionary, offering a hands-on example.


Data
----

To use Hy2DL, you will need to download and prepare the datasets. These are not included in the package. Instructions on how to download and prepare the datasets can be found in the `data folder <https://github.com/eduardoAcunaEspinoza/Hy2DL/tree/main/data>`_ of the repository.