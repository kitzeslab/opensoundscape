# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from m2r import MdInclude

# local import for linking to GitHub source code
from sphinx_linkcode import make_linkcode_resolve

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "opensoundscape"
copyright = "2022 Sam Lapp, Tessa Rhinehart, Louis Freeland-Haynes, Jatin Khilnani, Alexandra Syunkova, Justin Kitzes"
author = "Sam Lapp, Tessa Rhinehart, Louis Freeland-Haynes, Jatin Khilnani, Alexandra Syunkova, Justin Kitzes"

# The full version, including alpha/beta/rc tags
release = "0.9.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def setup(app):
    config = {
        # 'url_resolver': lambda url: github_doc_root + url,
        "auto_toc_tree_section": "Contents",
        "enable_eval_rst": True,
    }

    # from m2r to make `mdinclude` work
    app.add_config_value("no_underscore_emphasis", False, "env")
    app.add_config_value("m2r_parse_relative_links", False, "env")
    app.add_config_value("m2r_anonymous_references", False, "env")
    app.add_config_value("m2r_disable_inline_math", False, "env")
    app.add_directive("mdinclude", MdInclude)


# The following is used by sphinx.ext.linkcode to provide links to github
# implementation based on sklearn
linkcode_resolve = make_linkcode_resolve(
    "opensoundscape",
    (
        "https://github.com/kitzeslab/"
        "opensoundscape/blob/{revision}/"
        "{package}/{path}#L{lineno}"
    ),
)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_static_path = ["_static"]
html_css_files = ["css/greentheme.css"]

style_nav_header_background = "#2980B9"

# Mock libraries we don't want to install on RTD
autodoc_mock_imports = [
    "docopt",
    "pandas",
    "librosa",
    "ray",
    "torch",
    "sklearn",
    "numpy",
    "schema",
    "soundfile",
    "scipy",
    "yaml",
    "torchvision",
    "matplotlib",
    "pywt",
    "deprecated",
    "skimage",
    "wandb",
    "pytorch_grad_cam",
    "aru_metadata_parser",
    "pytz",
    "pillow",
    "PIL",
]

master_doc = "index"
