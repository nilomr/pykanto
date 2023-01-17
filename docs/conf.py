# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path
import re
from sphinx.util.inspect import safe_getattr
from docutils.parsers.rst import directives
from sphinx.ext.autosummary import get_documenter
from sphinx.ext.autosummary import Autosummary
import os
import sys

sys.path.insert(0, os.path.abspath("."))
# Reference
# https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/


# -- Project information -----------------------------------------------------

project = "pykanto"
copyright = "2021, Nilo M. Recalde"
author = "Nilo M. Recalde"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
]
nb_execution_mode = "cache"
coverage_show_missing_items = True
autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ["_templates"]
autodoc_member_order = "bysource"

# Strip input prompts from copied code
copybutton_prompt_text = ">>> "
copybutton_prompt_text = (
    r">>> |^\d+|\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/nilomr/pykanto",
    "use_repository_button": True,
    "logo_only": True,
    "extra_navbar": False,
}

# nb_execution_mode = "cache"
nb_execution_mode = "auto"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

html_logo = str(Path("custom") / "pykanto-logo-grey-04.svg")

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "member-order": "bysource",
}

default_role = "py:obj"
