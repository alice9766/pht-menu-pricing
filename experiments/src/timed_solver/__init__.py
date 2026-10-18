"""Phase-separated copies of the exact PHT recurrences for benchmarking.

The frozen solver snapshot remains unchanged. These helpers accept already
validated exact inputs and return deferred recovery functions. No buyer replay
is performed inside optimization or recovery.
"""
from .cores import optimize_joint, optimize_capacity
