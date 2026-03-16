"""Sphinx configuration for the AMPS code documentation."""

import sys
from pathlib import Path

# Make the code/ directory importable so autodoc can inspect modules.
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

# -- Project information -------------------------------------------------------
project = "AMPS"
author = "AMPS Research Team"
release = "0.1"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Avoid importing heavy runtime dependencies during the doc build.
autodoc_mock_imports = [
    "egret",
    "pyomo",
    "highspy",
    "gurobipy",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
]

autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "AMPS Documentation"
