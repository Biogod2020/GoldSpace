# spaglam_preproc/__init__.py

__version__ = "0.1.0"

# Expose the main user-facing classes and functions for easy import
# This allows users to write `from spaglam_preproc import SpaglamPipeline`
from .core.dataset_writer import SpaglamPipeline, create_dataset_shards
from .core.image_tiler import ImageHandler

__all__ = [
    "SpaglamPipeline",
    "create_dataset_shards",
    "ImageHandler",
    "__version__",
]
