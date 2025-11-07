"""
Model architectures for super-resolution.
"""

from .pannet import PanNet
from .srcnn import SRCNN
from .vdsr import VDSR
from .srgan import Generator, Discriminator

__all__ = ['PanNet', 'SRCNN', 'VDSR', 'Generator', 'Discriminator']

