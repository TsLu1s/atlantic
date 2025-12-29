from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from atlantic.core.enums import DimensionalityTier, OptimizationLevel


@dataclass
class DimensionalityConfig:
    """Configuration for optimization based on data dimensions."""
    
    tier: DimensionalityTier
    optimization_level: OptimizationLevel
    n_trials: int
    n_folds: int
    early_stopping_rounds: Optional[int]
    model_complexity: Literal["low", "medium", "high"]
    
    def __repr__(self) -> str:
        return (
            f"DimensionalityConfig(tier={self.tier.value}, "
            f"optimization_level={self.optimization_level.value}, "
            f"n_trials={self.n_trials}, n_folds={self.n_folds})"
        )


class DimensionalityStrategy:
    """
    Adaptive strategy for optimization parameters based on data shape.
    
    Determines optimal number of trials, folds, and model complexity
    based on dataset dimensions and user-specified optimization level.
    """
    
    # Tier thresholds: (max_rows, max_features)
    TIER_THRESHOLDS: Dict[DimensionalityTier, Tuple[int, int]] = {
        DimensionalityTier.SMALL: (5000, 30),
        DimensionalityTier.MEDIUM: (20000, 100),
        DimensionalityTier.LARGE: (100000, 500),
        DimensionalityTier.XLARGE: (float('inf'), float('inf'))
    }
    
    # Trial matrix: [tier][aggressiveness] -> n_trials
    TRIAL_MATRIX: Dict[DimensionalityTier, Dict[OptimizationLevel, int]] = {
        DimensionalityTier.SMALL: {
            OptimizationLevel.FAST: 5,
            OptimizationLevel.BALANCED: 10,
            OptimizationLevel.THOROUGH: 20
        },
        DimensionalityTier.MEDIUM: {
            OptimizationLevel.FAST: 4,
            OptimizationLevel.BALANCED: 8,
            OptimizationLevel.THOROUGH: 15
        },
        DimensionalityTier.LARGE: {
            OptimizationLevel.FAST: 3,
            OptimizationLevel.BALANCED: 6,
            OptimizationLevel.THOROUGH: 10
        },
        DimensionalityTier.XLARGE: {
            OptimizationLevel.FAST: 2,
            OptimizationLevel.BALANCED: 4,
            OptimizationLevel.THOROUGH: 7
        }
    }
    
    # Fold configuration by tier
    FOLD_CONFIG: Dict[DimensionalityTier, int] = {
        DimensionalityTier.SMALL: 5,
        DimensionalityTier.MEDIUM: 4,
        DimensionalityTier.LARGE: 3,
        DimensionalityTier.XLARGE: 3
    }
    
    # Model complexity by tier
    COMPLEXITY_CONFIG: Dict[DimensionalityTier, str] = {
        DimensionalityTier.SMALL: "high",
        DimensionalityTier.MEDIUM: "medium",
        DimensionalityTier.LARGE: "medium",
        DimensionalityTier.XLARGE: "low"
    }
    
    # Early stopping configuration
    EARLY_STOPPING: Dict[DimensionalityTier, Optional[int]] = {
        DimensionalityTier.SMALL: None,
        DimensionalityTier.MEDIUM: 5,
        DimensionalityTier.LARGE: 3,
        DimensionalityTier.XLARGE: 2
    }
    
    @classmethod
    def detect_tier(cls, n_rows: int, n_features: int) -> DimensionalityTier:
        """
        Detect dimensionality tier based on data shape.

        Returns:
            Detected DimensionalityTier.
        """
        for tier, (max_rows, max_features) in cls.TIER_THRESHOLDS.items():
            if n_rows < max_rows and n_features < max_features:
                return tier
        
        return DimensionalityTier.XLARGE
    
    @classmethod
    def get_config(
        cls,
        n_rows: int,
        n_features: int,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    ) -> DimensionalityConfig:
        """
        Get optimization configuration for given data dimensions.

        Returns:
            DimensionalityConfig with optimal settings.
        """
        tier = cls.detect_tier(n_rows, n_features)
        
        return DimensionalityConfig(
            tier=tier,
            optimization_level=optimization_level,
            n_trials=cls.TRIAL_MATRIX[tier][optimization_level],
            n_folds=cls.FOLD_CONFIG[tier],
            early_stopping_rounds=cls.EARLY_STOPPING[tier],
            model_complexity=cls.COMPLEXITY_CONFIG[tier]
        )
    
    @classmethod
    def get_config_from_string(
        cls,
        n_rows: int,
        n_features: int,
        optimization_level: str = "balanced"
    ) -> DimensionalityConfig:
        """
        Get configuration using string optimization level.

        Returns:
            DimensionalityConfig with optimal settings.
        """
        agg_map = {
            "fast": OptimizationLevel.FAST,
            "balanced": OptimizationLevel.BALANCED,
            "thorough": OptimizationLevel.THOROUGH
        }
        
        agg_level = agg_map.get(optimization_level.lower(), OptimizationLevel.BALANCED)
        return cls.get_config(n_rows, n_features, agg_level)
    
    @classmethod
    def describe_tier(cls, tier: DimensionalityTier) -> str:
        """
        Get human-readable description of tier.

        Returns:
            Description string.
        """
        descriptions = {
            DimensionalityTier.SMALL: "Small dataset (< 5K rows, < 30 features)",
            DimensionalityTier.MEDIUM: "Medium dataset (< 20K rows, < 100 features)",
            DimensionalityTier.LARGE: "Large dataset (< 100K rows, < 500 features)",
            DimensionalityTier.XLARGE: "Extra-large dataset (>= 100K rows or >= 500 features)"
        }
        return descriptions.get(tier, "Unknown tier")


@dataclass
class ModelHyperparameterRanges:
    """Hyperparameter ranges adjusted by model complexity."""
    
    # Random Forest / Extra Trees
    rf_n_estimators: Tuple[int, int]
    rf_max_depth: Tuple[int, int]
    rf_min_samples_split: Tuple[int, int]
    
    # XGBoost
    xgb_n_estimators: Tuple[int, int]
    xgb_max_depth: Tuple[int, int]
    xgb_learning_rate: Tuple[float, float]
    
    @classmethod
    def for_complexity(cls, complexity: str) -> "ModelHyperparameterRanges":
        """
        Get hyperparameter ranges for given complexity level.

        Returns:
            ModelHyperparameterRanges instance.
        """
        if complexity == "low":
            return cls(
                rf_n_estimators=(30, 100),
                rf_max_depth=(3, 15),
                rf_min_samples_split=(2, 15),
                xgb_n_estimators=(30, 100),
                xgb_max_depth=(3, 12),
                xgb_learning_rate=(0.05, 0.2)
            )
        elif complexity == "medium":
            return cls(
                rf_n_estimators=(50, 150),
                rf_max_depth=(5, 25),
                rf_min_samples_split=(2, 20),
                xgb_n_estimators=(50, 150),
                xgb_max_depth=(5, 18),
                xgb_learning_rate=(0.03, 0.15)
            )
        else:  # high
            return cls(
                rf_n_estimators=(50, 200),
                rf_max_depth=(5, 32),
                rf_min_samples_split=(2, 25),
                xgb_n_estimators=(50, 200),
                xgb_max_depth=(5, 25),
                xgb_learning_rate=(0.01, 0.1)
            )