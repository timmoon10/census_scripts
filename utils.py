import functools
import pathlib

@functools.cache
def root_dir() -> pathlib.Path:
    """Path to project root directory"""
    return pathlib.Path(__file__).resolve().parent

def data_dir() -> pathlib.Path:
    """Path to project data directory"""
    return root_dir() / "data"

def results_dir() -> pathlib.Path:
    """Path to project results directory"""
    return root_dir() / "results"
