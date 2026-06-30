Head layer
=============

:py:class:`hy2dl.modelzoo.head` present classes to map the hidden states produced by the LSTM into model outputs. The head layer is defined by the configuration key ``head``.

Regression
-----------------
Fully connected layer that maps the LSTM hidden states to output features. Output features is set to len(``target``)  by default, but can the ajusted using the configuration key ``output_features``.


Mix-Density network (MDN)
-----------------------------------
Defines a mix-density network (MDN) output layer, enabling probabilistic predictions. Further details can be found in :ref:`mdn_reference` and `hy2dl.utils.distributions`.