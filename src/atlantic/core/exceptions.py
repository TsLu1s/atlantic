"""
Atlantic - Exception Hierarchy.

Every exception raised deliberately by the package derives from
:class:`AtlanticError`, so ``except AtlanticError`` catches all of them.

``ValidationError`` and ``NotFittedError`` additionally derive from the
built-in ``ValueError``. This mirrors scikit-learn / pydantic conventions and
keeps existing ``except ValueError`` handlers working.
"""

from typing import Optional


class AtlanticError(Exception):
    """Base exception for Atlantic package."""


class NotFittedError(AtlanticError, ValueError):
    """Raised when transform is called before fit."""

    def __init__(self, component: str = "Pipeline", message: Optional[str] = None):
        self.component = component
        super().__init__(message or f"{component} has not been fitted. Call fit() first.")


class ValidationError(AtlanticError, ValueError):
    """Raised when input validation fails."""


class FeatureSelectionError(AtlanticError):
    """Raised when feature selection fails."""


class EncodingError(AtlanticError):
    """Raised when encoding fails."""


class ImputationError(AtlanticError):
    """Raised when imputation fails."""


class RegistryError(AtlanticError):
    """Raised when component registry operations fail."""


class ConfigurationError(AtlanticError):
    """Raised when configuration is invalid."""


class H2OConnectionError(FeatureSelectionError):
    """Raised when H2O is unavailable or its server connection fails.

    Subclasses :class:`FeatureSelectionError` because the H2O cluster is only
    used for feature selection - handlers catching the broader error keep
    working.
    """


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails during transform."""


class StateSerializationError(AtlanticError):
    """Raised when pipeline state serialization/deserialization fails."""
