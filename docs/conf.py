"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "patch-denoise"
copyright = "2022, Pierre-Antoine Comby"
author = "Pierre-Antoine Comby"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "cli_generated.rst"]

_python_doc_base = "https://docs.python.org/3.9"

intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# generate autosummary even if no references
autosummary_generate = True
# autosummary_imported_members = True
autodoc_inherit_docstrings = True

napoleon_include_private_with_doc = True

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "filename_pattern": "/example_",
    "ignore_pattern": "conftest.py",
    "example_extensions": {".py"},
    "gallery_dirs": ["auto_examples"],
    "reference_url": {
        "numpy": "http://docs.scipy.org/doc/numpy-1.9.1",
        "scipy": "http://docs.scipy.org/doc/scipy-0.17.0/reference",
    },
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
html_context = {"default_mode": "light"}


def _generate_cli_docs(app):
    """Render the cyclopts CLI app help as RST, included from usage.rst."""
    from patch_denoise.bindings.cli import app as cli_app

    rst = cli_app.generate_docs(output_format="rst", heading_level=2)
    out_path = os.path.join(app.srcdir, "cli_generated.rst")
    with open(out_path, "w") as f:
        f.write(rst)


def setup(app):
    """Register the CLI doc generation hook on the Sphinx build."""
    app.connect("builder-inited", _generate_cli_docs)
