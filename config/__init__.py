"""
Configuration Management for Market Dashboard

Provides centralized access to all configurable parameters.
Load once at startup, access anywhere via `from config import cfg`

Usage:
    from config import cfg

    # Access nested values
    ema_period = cfg.credit.left_strategy.ema_period
    vvix_buy = cfg.volatility.vvix.strong_buy_threshold

    # Or use get() for safe access with defaults
    lookback = cfg.get('data_collection.lookback_days.short', 90)
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigSection:
    """
    A configuration section that allows dot-notation access.

    Example:
        section = ConfigSection({'foo': {'bar': 42}})
        section.foo.bar  # Returns 42
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"ConfigSection({self.__dict__})"

    def to_dict(self) -> dict:
        """Convert back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigSection):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class Config:
    """
    Main configuration class for Market Dashboard.

    Loads parameters from config/parameters.yaml and provides
    convenient access via dot notation or get() method.
    """

    _instance: Optional['Config'] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern - only one config instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._initialized:
            return

        self._data: dict = {}
        self._load_config()
        Config._initialized = True

    def _load_config(self):
        """Load configuration from YAML file"""
        # Find config file relative to this module
        config_dir = Path(__file__).parent
        config_path = config_dir / "parameters.yaml"

        # Fallback to project root if not found
        if not config_path.exists():
            project_root = config_dir.parent
            config_path = project_root / "config" / "parameters.yaml"

        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._data = self._get_defaults()
        else:
            try:
                with open(config_path, 'r') as f:
                    self._data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self._data = self._get_defaults()

        # Convert to dot-notation accessible sections
        for key, value in self._data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)

    def _get_defaults(self) -> dict:
        """Return minimal default configuration"""
        return {
            'volatility': {
                'vrp': {'lookback_days': 21},
                'vvix': {
                    'strong_buy_threshold': 120,
                    'buy_alert_threshold': 110,
                    'normal_min': 80,
                },
            },
            'credit': {
                'left_strategy': {
                    'ema_period': 330,
                    'entry_threshold': 0.65,
                    'exit_threshold': 1.40,
                }
            },
            'liquidity': {
                'net_liquidity': {
                    'lookback_days': 252,
                    'supportive_threshold': 0.8,
                    'draining_threshold': -0.8,
                }
            },
            'data_collection': {
                'lookback_days': {
                    'short': 90,
                    'medium': 365,
                    'long': 730,
                },
                'retry': {
                    'max_retries': 3,
                    'initial_delay': 1.0,
                    'max_delay': 60.0,
                    'backoff_multiplier': 2.0,
                }
            },
        }

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.

        Args:
            path: Dot-separated path (e.g., 'volatility.vvix.strong_buy_threshold')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self._data

        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def reload(self):
        """Reload configuration from file"""
        Config._initialized = False
        self.__init__()

    def to_dict(self) -> dict:
        """Return full configuration as dictionary"""
        return self._data.copy()


# Global config instance - import this
cfg = Config()


# Convenience functions for common access patterns
def get_lookback(period: str = 'medium') -> int:
    """Get standard lookback period in days"""
    return cfg.get(f'data_collection.lookback_days.{period}', 365)


def get_color(color_type: str) -> str:
    """Get standard color by type (positive, negative, neutral, etc.)"""
    return cfg.get(f'display.colors.{color_type}', '#9E9E9E')
