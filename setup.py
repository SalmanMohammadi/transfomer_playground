
from distutils.core import setup

setup(name='transformer_playground',
      install_requires = [
          'torch',
          'numpy',
          'einops',
          'matplotlib',
          'torchvision',
          'black',
          'pycodestyle'
      ]
     )