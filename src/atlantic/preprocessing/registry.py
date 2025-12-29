from __future__ import annotations

from typing import Any, Dict, Generic, List, Type, TypeVar

from atlantic.core.exceptions import RegistryError

T = TypeVar('T')


class ComponentRegistry(Generic[T]):
    """Generic registry for preprocessing components."""
    
    def __init__(self, component_type: str):
        """
        Initialize registry.
        
        Args:
            component_type: Type name for error messages.
        """
        self._components: Dict[str, Type[T]] = {}
        self._component_type = component_type
    
    def register(self, name: str, component: Type[T]) -> None:
        """
        Register a component class.
        
        Args:
            name: Identifier for the component.
            component: Component class to register.
        """
        self._components[name.lower()] = component
    
    def get(self, name: str, **kwargs) -> T:
        """
        Get an instantiated component by name.

        Raises:
            RegistryError: If component not found.
        """
        name_lower = name.lower()
        if name_lower not in self._components:
            available = self.list_available()
            raise RegistryError(
                f"{self._component_type} '{name}' not registered. "
                f"Available: {available}"
            )
        return self._components[name_lower](**kwargs)
    
    def get_class(self, name: str) -> Type[T]:
        """
        Get component class without instantiation.

        Raises:
            RegistryError: If component not found.
        """
        name_lower = name.lower()
        if name_lower not in self._components:
            available = self.list_available()
            raise RegistryError(
                f"{self._component_type} '{name}' not registered. "
                f"Available: {available}"
            )
        return self._components[name_lower]
    
    def list_available(self) -> List[str]:
        """
        Get list of registered component names.
        
        Returns:
            List of component identifiers.
        """
        return list(self._components.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if component is registered.

        Returns:
            True if registered, False otherwise.
        """
        return name.lower() in self._components


# Import components for registration
from atlantic.preprocessing.encoders import (
    AutoLabelEncoder,
    AutoOneHotEncoder,
    AutoIFrequencyEncoder
)
from atlantic.preprocessing.scalers import (
    AutoMinMaxScaler,
    AutoStandardScaler,
    AutoRobustScaler
)
from atlantic.preprocessing.imputers import (
    AutoSimpleImputer,
    AutoKNNImputer,
    AutoIterativeImputer
)


# Create registry instances
EncoderRegistry = ComponentRegistry[Any]("Encoder")
ScalerRegistry = ComponentRegistry[Any]("Scaler")
ImputerRegistry = ComponentRegistry[Any]("Imputer")


# Register encoders
EncoderRegistry.register("label", AutoLabelEncoder)
EncoderRegistry.register("ifrequency", AutoIFrequencyEncoder)
EncoderRegistry.register("onehot", AutoOneHotEncoder)

# Register scalers
ScalerRegistry.register("standard", AutoStandardScaler)
ScalerRegistry.register("minmax", AutoMinMaxScaler)
ScalerRegistry.register("robust", AutoRobustScaler)

# Register imputers
ImputerRegistry.register("simple", AutoSimpleImputer)
ImputerRegistry.register("knn", AutoKNNImputer)
ImputerRegistry.register("iterative", AutoIterativeImputer)