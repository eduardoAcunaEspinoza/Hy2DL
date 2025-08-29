# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "Hy2DL"
copyright = "2025, Eduardo Acuna"
author = "Eduardo Acuna"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # autodocument
    "sphinx.ext.napoleon",  # google and numpy doc string support
    "sphinx.ext.mathjax",  # latex rendering of equations using MathJax
    "nbsphinx",  # for direct embedding of jupyter notebooks into sphinx docs
    #'nbsphinx_link'  # to be able to include notebooks from outside of the docs folder
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/Hy2DL.png"
