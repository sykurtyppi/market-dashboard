"""
Data collectors package
"""

from .fred_collector import FREDCollector
from .fear_greed_collector import FearGreedCollector
from .cboe_collector import CBOECollector
# Breadth collector replaced with SP500ADLineCalculator
from .sp500_adline_calculator import SP500ADLineCalculator
from .yahoo_collector import YahooCollector

__all__ = [
    'FREDCollector',
    'FearGreedCollector',
    'CBOECollector',
    'SP500ADLineCalculator',
    'YahooCollector']
