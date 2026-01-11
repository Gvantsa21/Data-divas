"""
Spotify Track Analysis Package

This package contains modules for data processing, visualization, and machine learning
for analyzing Spotify track data and predicting popularity.

Modules:
    data_processing: Functions for data cleaning and preprocessing
    visualization: Functions for creating visualizations and plots
    models: Machine learning model implementations and evaluation
"""

__version__ = "1.0.0"
__author__ = "Gvantsa Tchuradze, Mariam Phirtskhalava, Ani Kharabadze"

from . import data_processing
from . import visualization
from . import models

__all__ = ['data_processing', 'visualization', 'models']