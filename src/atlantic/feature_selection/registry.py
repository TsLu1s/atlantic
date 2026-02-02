from __future__ import annotations

from typing import Dict, List, Type

from atlantic.core.exceptions import RegistryError
from atlantic.feature_selection.h2o_selector import H2OFeatureSelector
from atlantic.feature_selection.vif_selector import VIFFeatureSelector
from atlantic.preprocessing.base import BaseFeatureSelector


class FeatureSelectorRegistry:
    """
    Registry for feature selector components.
    """

    _selectors: Dict[str, Type[BaseFeatureSelector]] = {}

    @classmethod
    def register(cls, name: str, selector: Type[BaseFeatureSelector]) -> None:
        """
        Register a feature selector class.

        Args:
            name: Identifier for the selector.
            selector: Selector class to register.
        """
        cls._selectors[name.lower()] = selector

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseFeatureSelector:
        """
        Get an instantiated selector by name.

        Raises:
            RegistryError: If selector not found.
        """
        name_lower = name.lower()
        if name_lower not in cls._selectors:
            available = cls.list_available()
            raise RegistryError(
                f"Feature selector '{name}' not registered. " f"Available: {available}"
            )
        return cls._selectors[name_lower](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[BaseFeatureSelector]:
        """
        Get selector class without instantiation.

        Returns:
            Selector class.
        """
        name_lower = name.lower()
        if name_lower not in cls._selectors:
            available = cls.list_available()
            raise RegistryError(
                f"Feature selector '{name}' not registered. " f"Available: {available}"
            )
        return cls._selectors[name_lower]

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Get list of registered selector names.
        """
        return list(cls._selectors.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if selector is registered.
        """
        return name.lower() in cls._selectors


FeatureSelectorRegistry.register("h2o", H2OFeatureSelector)
FeatureSelectorRegistry.register("vif", VIFFeatureSelector)
