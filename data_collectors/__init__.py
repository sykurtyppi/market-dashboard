"""
Data collectors package
"""

from .fred_collector import FREDCollector
from .fear_greed_collector import FearGreedCollector
from .cboe_collector import CBOECollector
# Breadth collector replaced with SP500ADLineCalculator
from .sp500_adline_calculator import SP500ADLineCalculator
from .yahoo_collector import YahooCollector
from .cta_collector import CTACollector
from .insider_trading_collector import InsiderTradingCollector
from .dark_pool_collector import DarkPoolCollector
from .economic_calendar_collector import EconomicCalendarCollector

__all__ = [
    'FREDCollector',
    'FearGreedCollector',
    'CBOECollector',
    'SP500ADLineCalculator',
    'YahooCollector',
    'CTACollector',
    'InsiderTradingCollector',
    'DarkPoolCollector',
    'EconomicCalendarCollector',
]
