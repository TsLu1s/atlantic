class AtlanticError(Exception):
    """Base exception for Atlantic package."""
    pass


class NotFittedError(AtlanticError):
    """Raised when transform is called before fit."""
    
    def __init__(self, component: str = "Pipeline"):
        self.component = component
        super().__init__(f"{component} has not been fitted. Call fit() first.")


class ValidationError(AtlanticError):
    """Raised when input validation fails."""
    pass


class FeatureSelectionError(AtlanticError):
    """Raised when feature selection fails."""
    pass


class EncodingError(AtlanticError):
    """Raised when encoding fails."""
    pass


class ImputationError(AtlanticError):
    """Raised when imputation fails."""
    pass


class RegistryError(AtlanticError):
    """Raised when component registry operations fail."""
    pass


class ConfigurationError(AtlanticError):
    """Raised when configuration is invalid."""
    pass


class H2OConnectionError(AtlanticError):
    """Raised when H2O server connection fails."""
    pass


class SchemaValidationError(AtlanticError):
    """Raised when schema validation fails during transform."""
    pass


class StateSerializationError(AtlanticError):
    """Raised when pipeline state serialization/deserialization fails."""
    pass