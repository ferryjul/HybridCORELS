from __future__ import print_function, division, with_statement
from .HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from .utils import load_from_csv
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')) as f:
    __version__ = f.read().strip()

__all__ = ["HybridCORELSPreClassifier", "HybridCORELSPostClassifier", "load_from_csv"]
