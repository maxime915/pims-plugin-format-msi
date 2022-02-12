"setup: package information"

import io
import os

from setuptools import setup, find_packages

# Package meta-data
NAME = 'pims-plugin-format-msi'
REQUIRES_PYTHON = '>=3.8.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'pims',
    'zarr',
    'lxml',
    'pyimzml',
    'numpy',
    'pandas',
]

DEPENDENCY_LINKS = []

# What packages are optional?
EXTRAS = {
    'tests': ['pytest>=6.2.2'],
}

# Load the package's __version__.py module as a directory
about = {}
here = os.path.abspath(os.path.dirname(__file__))
project_slug = NAME.lower().replace('-', '_').replace(' ', '_')
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = about['__description__']

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__email__'],
    python_requires=REQUIRES_PYTHON,
    url=about['__url__'],
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    entry_points={
        'pims.formats': f'{about["__plugin__"]} = {project_slug}',
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    include_package_data=True,
    license=about['__license__'],
)
