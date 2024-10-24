from pathlib import Path
from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")

# read the description from the README.md
def readme():
    with open("README.md") as f:
        return f.read().strip()

def version():
    with open("hy2dl/__version__.py") as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc["__version__"]
    
setup(name='Hy2DL',
      license="GPL-3.0",
      version=version(),
      author='Eduardo Acuña Espinoza, Ralf Loritz, Manuel Álvarez Chaves',
      author_email='eduardo.espinoza@kit.edu',
      description='Library to create hydrological models for rainfall-runoff prediction, using deep learning methods',
      long_description=readme(),
      long_description_content_type='text/markdown',
      python_requires='>=3.8',
      install_requires=requirements(),
      packages=find_packages(),
      keywords='deep learning hydrology lstm neural network streamflow discharge rainfall-runoff')
