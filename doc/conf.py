# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../falkon'))

# Need mocking to allow everything to be imported even on no-GPU machines
autodoc_mock_imports = [
    # "torch",
    # "pykeops",
    # "numpy",
    "falkon.cuda",
    "falkon.ooc_ops.multigpu_potrf"
]

# -- Project information -----------------------------------------------------

project = 'falkon'
copyright = '2020, Giacomo Meanti, Alessandro Rudi'
author = 'Giacomo Meanti, Alessandro Rudi'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'numpydoc',  # Our docs are numpy style
    'sphinx_rtd_theme',  # Read-the-docs theme
    'sphinx.ext.mathjax',  # For displaying math in html output
    'nbsphinx',  # For displaying jupyter notebooks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# General information about the project.
project = u'falkon'
copyright = u'2020, Giacomo Meanti'
author = u'Giacomo Meanti, Alessandro Rudi'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
def get_version(root_dir):
    with open(os.path.join(root_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version

# The short X.Y version.
version = get_version("..")
# The full version, including alpha/beta/rc tags.
release = version

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Output file base name for HTML help builder.
htmlhelp_basename = 'falkondoc'

numpydoc_show_class_members = False


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
#     'numpy': ('https://docs.scipy.org/doc/numpy/', None),
#     'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
#     'matplotlib': ('https://matplotlib.org/', None),
#     'sklearn': ('http://scikit-learn.org/stable', None),
# }

# sphinx_gallery_conf = {
#     'backreferences_dir': 'gen_modules/backreferences',
#     'doc_module': ('celer', 'numpy'),
#     'examples_dirs': '../examples',
#     'gallery_dirs': 'auto_examples',
#     'reference_url': {
#         'celer': None,
#     }
# }

html_sidebars = {'**': ['globaltoc.html', 'localtoc.html', 'searchbox.html']}