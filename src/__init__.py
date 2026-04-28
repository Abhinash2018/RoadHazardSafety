"""
Road Hazard Detection System - Multi-Sensor Data Fusion Framework
"""

__version__ = "1.0.0"
__author__ = "Abinash Ghimire"

from . import config
from . import data_acquisition
from . import feature_extraction
from . import model_trainer
from . import evaluator

__all__ = [
    "config",
    "data_acquisition", 
    "feature_extraction",
    "model_trainer",
    "evaluator",
]
