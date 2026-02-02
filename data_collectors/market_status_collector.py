"""
Market Status Collector
Tracks market hours, pre/post market, and holiday schedule

Provides:
- Current market status (Open/Closed/Pre-Market/After-Hours)
- Time until market opens/closes
- Holiday calendar
- Trading session info
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Market Holidays 2024-2026 (NYSE/NASDAQ closed)
US_MARKET_HOLIDAYS = {
    # 2024
    datetime(2024, 1, 1): "New Year's Day",
    datetime(2024, 1, 15): "MLK Day",
    datetime(2024, 2, 19): "Presidents Day",
    datetime(2024, 3, 29): "Good Friday",
    datetime(2024, 5, 27): "Memorial Day",
    datetime(2024, 6, 19): "Juneteenth",
    datetime(2024, 7, 4): "Independence Day",
    datetime(2024, 9, 2): "Labor Day",
    datetime(2024, 11, 28): "Thanksgiving",
    datetime(2024, 12, 25): "Christmas",
    # 2025
    datetime(2025, 1, 1): "New Year's Day",
    datetime(2025, 1, 20): "MLK Day",
    datetime(2025, 2, 17): "Presidents Day",
    datetime(2025, 4, 18): "Good Friday",
    datetime(2025, 5, 26): "Memorial Day",
    datetime(2025, 6, 19): "Juneteenth",
    datetime(2025, 7, 4): "Independence Day",
    datetime(2025, 9, 1): "Labor Day",
    datetime(2025, 11, 27): "Thanksgiving",
    datetime(2025, 12, 25): "Christmas",
    # 2026
    datetime(2026, 1, 1): "New Year's Day",
    datetime(2026, 1, 19): "MLK Day",
    datetime(2026, 2, 16): "Presidents Day",
    datetime(2026, 4, 3): "Good Friday",
    datetime(2026, 5, 25): "Memorial Day",
    datetime(2026, 6, 19): "Juneteenth",
    datetime(2026, 7, 3): "Independence Day (observed)",
    datetime(2026, 9, 7): "Labor Day",
    datetime(2026, 11, 26): "Thanksgiving",
    datetime(2026, 12, 25): "Christmas",
}

# Early close days (1:00 PM ET) - day before/after holidays
EARLY_CLOSE_DAYS = {
    datetime(2024, 7, 3): "Day before Independence Day",
    datetime(2024, 11, 29): "Day after Thanksgiving",
    datetime(2024, 12, 24): "Christmas Eve",
    datetime(2025, 7, 3): "Day before Independence Day",
    datetime(2025, 11, 28): "Day after Thanksgiving",
    datetime(2025, 12, 24): "Christmas Eve",
    datetime(2026, 11, 27): "Day after Thanksgiving",
    datetime(2026, 12, 24): "Christmas Eve",
}


class MarketStatusCollector:
    """Tracks US stock market hours and status"""

    def __init__(self):
        self.eastern = ZoneInfo("America/New_York")

        # Market hours (Eastern Time)
        self.pre_market_start = time(4, 0)   # 4:00 AM ET
        self.market_open = time(9, 30)        # 9:30 AM ET
        self.market_close = time(16, 0)       # 4:00 PM ET
        self.after_hours_end = time(20, 0)    # 8:00 PM ET
        self.early_close = time(13, 0)        # 1:00 PM ET (early close days)

    def get_market_status(self) -> Dict:
        """
        Get current market status

        Returns:
            Dict with status, time info, and next session
        """
        now_et = datetime.now(self.eastern)
        today = now_et.date()
        current_time = now_et.time()

        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_open = self._get_next_market_open(now_et)
            return {
                'status': 'CLOSED',
                'status_color': '#F44336',
                'reason': 'Weekend',
                'emoji': '🌙',
                'next_open': next_open,
                'time_until_open': self._format_timedelta(next_open - now_et) if next_open else None,
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        # Check if holiday
        today_dt = datetime(today.year, today.month, today.day)
        if today_dt in US_MARKET_HOLIDAYS:
            next_open = self._get_next_market_open(now_et)
            return {
                'status': 'CLOSED',
                'status_color': '#F44336',
                'reason': f"Holiday: {US_MARKET_HOLIDAYS[today_dt]}",
                'emoji': '🎉',
                'next_open': next_open,
                'time_until_open': self._format_timedelta(next_open - now_et) if next_open else None,
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        # Check for early close
        is_early_close = today_dt in EARLY_CLOSE_DAYS
        close_time = self.early_close if is_early_close else self.market_close

        # Determine session
        if current_time < self.pre_market_start:
            # Before pre-market
            return {
                'status': 'CLOSED',
                'status_color': '#9E9E9E',
                'reason': 'Overnight',
                'emoji': '🌙',
                'next_session': 'Pre-Market at 4:00 AM ET',
                'time_until_premarket': self._time_until(self.pre_market_start, now_et),
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        elif current_time < self.market_open:
            # Pre-market
            return {
                'status': 'PRE-MARKET',
                'status_color': '#FF9800',
                'reason': 'Extended hours trading',
                'emoji': '🌅',
                'next_session': 'Market Opens at 9:30 AM ET',
                'time_until_open': self._time_until(self.market_open, now_et),
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        elif current_time < close_time:
            # Market open
            time_remaining = self._time_until(close_time, now_et)
            early_note = " (Early Close)" if is_early_close else ""
            return {
                'status': 'OPEN',
                'status_color': '#4CAF50',
                'reason': f'Regular trading session{early_note}',
                'emoji': '📈',
                'closes_at': close_time.strftime('%I:%M %p ET'),
                'time_until_close': time_remaining,
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        elif current_time < self.after_hours_end:
            # After hours
            return {
                'status': 'AFTER-HOURS',
                'status_color': '#2196F3',
                'reason': 'Extended hours trading',
                'emoji': '🌆',
                'ends_at': '8:00 PM ET',
                'time_until_end': self._time_until(self.after_hours_end, now_et),
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

        else:
            # After after-hours
            next_open = self._get_next_market_open(now_et)
            return {
                'status': 'CLOSED',
                'status_color': '#9E9E9E',
                'reason': 'Market closed for the day',
                'emoji': '🌙',
                'next_open': next_open,
                'time_until_open': self._format_timedelta(next_open - now_et) if next_open else None,
                'current_time_et': now_et.strftime('%I:%M %p ET'),
            }

    def _get_next_market_open(self, from_dt: datetime) -> Optional[datetime]:
        """Find next market open datetime"""
        next_day = from_dt + timedelta(days=1)

        # Look up to 10 days ahead (handles long weekends)
        for _ in range(10):
            next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)

            # Skip weekends
            if next_day.weekday() >= 5:
                next_day += timedelta(days=1)
                continue

            # Skip holidays
            day_key = datetime(next_day.year, next_day.month, next_day.day)
            if day_key in US_MARKET_HOLIDAYS:
                next_day += timedelta(days=1)
                continue

            return next_day

        return None

    def _time_until(self, target_time: time, from_dt: datetime) -> str:
        """Calculate time until a specific time today"""
        target_dt = from_dt.replace(
            hour=target_time.hour,
            minute=target_time.minute,
            second=0,
            microsecond=0
        )
        delta = target_dt - from_dt
        return self._format_timedelta(delta)

    def _format_timedelta(self, delta: timedelta) -> str:
        """Format timedelta as human-readable string"""
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "Now"

        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def get_next_holiday(self) -> Optional[Dict]:
        """Get the next upcoming market holiday"""
        now = datetime.now()
        today = datetime(now.year, now.month, now.day)

        for holiday_date, holiday_name in sorted(US_MARKET_HOLIDAYS.items()):
            if holiday_date >= today:
                days_away = (holiday_date - today).days
                return {
                    'date': holiday_date,
                    'name': holiday_name,
                    'days_away': days_away,
                }
        return None

    def is_market_open(self) -> bool:
        """Simple check if market is currently open"""
        status = self.get_market_status()
        return status['status'] == 'OPEN'

    def get_trading_sessions_today(self) -> Dict:
        """Get all trading session times for today"""
        now_et = datetime.now(self.eastern)
        today_dt = datetime(now_et.year, now_et.month, now_et.day)

        is_early_close = today_dt in EARLY_CLOSE_DAYS

        return {
            'pre_market': '4:00 AM - 9:30 AM ET',
            'regular': f"9:30 AM - {'1:00 PM' if is_early_close else '4:00 PM'} ET",
            'after_hours': f"{'1:00 PM' if is_early_close else '4:00 PM'} - 8:00 PM ET",
            'is_early_close': is_early_close,
            'early_close_reason': EARLY_CLOSE_DAYS.get(today_dt, None),
        }
