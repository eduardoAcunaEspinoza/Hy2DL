from pathlib import Path
from setuptools import setup

def requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")

# read the description from the README.md
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()

setup(name='Hy2DL',
      version="2.0",
      author='Eduardo Acuña Espinoza, Ralf Loritz, Manuel Álvarez Chaves',
      author_email='eduardo.espinoza@kit.edu',
      description='Library to create hydrological models for rainfall-runoff prediction, using deep learning methods',
      long_description=long_description,
      long_description_content_type='text/markdown',
      python_requires='>=3.8',
      install_requires=requirements(),
      keywords='deep learning hydrology lstm neural network streamflow discharge rainfall-runoff')
