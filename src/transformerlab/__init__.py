"""transformerlab: from-scratch transformer architectures with pluggable attention."""

__version__ = "0.1.0"

# Importing the attention subpackage registers every attention variant by name.
from . import attention  # noqa: F401

__all__ = ["attention", "__version__"]
