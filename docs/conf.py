# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re


def get_version_from_pyproject():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "r") as f:
        pyproject_content = f.read()
    version_match = re.search(r'^version\s*=\s*"(.*?)"', pyproject_content, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Version not found in pyproject.toml")


project = "DistClassiPy"
copyright = "2024, Siddharth Chaini"
author = "Siddharth Chaini"
release = get_version_from_pyproject()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/sidchaini/DistClassiPy",
    "use_repository_button": True,
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo-dark.png",
    },
}
