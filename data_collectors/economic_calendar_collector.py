"""
Economic Calendar Collector
Tracks major economic events and provides countdown to key releases

Uses multiple free data sources:
- FRED for historical economic data (CPI, NFP, GDP, PCE, Retail Sales)
- Hardcoded FOMC schedule (published annually by Fed)
- Calculated release dates (follows predictable government schedules)

FRED Series IDs for Economic Data:
- CPI: CPIAUCSL (Consumer Price Index - All Urban Consumers)
- NFP: PAYEMS (Total Nonfarm Payrolls, thousands)
- GDP: GDP (Gross Domestic Product, billions)
- PCE: PCEPI (Personal Consumption Expenditures Price Index)
- Retail Sales: RSAFS (Advance Retail Sales: Retail Trade)
- Unemployment: UNRATE (Unemployment Rate)

Each indicator shows actual, previous, and calculates YoY change where applicable.
"""

import logging
import os
import sys
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.secrets_helper import get_secret

logger = logging.getLogger(__name__)

# FRED Series IDs for economic indicators
FRED_SERIES = {
    'CPI': 'CPIAUCSL',           # Consumer Price Index
    'NFP': 'PAYEMS',             # Nonfarm Payrolls (thousands)
    'GDP': 'GDP',                # GDP (billions)
    'PCE': 'PCEPI',              # PCE Price Index
    'RETAIL': 'RSAFS',           # Retail Sales (millions)
    'UNEMPLOYMENT': 'UNRATE',    # Unemployment Rate
    'FED_FUNDS': 'FEDFUNDS',     # Fed Funds Rate
}


class EventImportance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EconomicEvent:
    """Represents a scheduled economic event"""
    name: str
    date: datetime
    importance: EventImportance
    category: str
    description: str = ""
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    yoy_change: Optional[float] = None  # Year-over-year % change
    unit: str = ""  # e.g., "%", "K jobs", "billions"
    fred_series: Optional[str] = None  # FRED series ID for data lookup


# FOMC Meeting Dates 2024-2025 (publicly announced)
FOMC_DATES_2024 = [
    datetime(2024, 1, 31),
    datetime(2024, 3, 20),
    datetime(2024, 5, 1),
    datetime(2024, 6, 12),
    datetime(2024, 7, 31),
    datetime(2024, 9, 18),
    datetime(2024, 11, 7),
    datetime(2024, 12, 18),
]

FOMC_DATES_2025 = [
    datetime(2025, 1, 29),
    datetime(2025, 3, 19),
    datetime(2025, 4, 30),  # May 1
    datetime(2025, 6, 18),
    datetime(2025, 7, 30),
    datetime(2025, 9, 17),
    datetime(2025, 11, 5),
    datetime(2025, 12, 17),
]

FOMC_DATES_2026 = [
    datetime(2026, 1, 28),
    datetime(2026, 3, 18),
    datetime(2026, 4, 29),
    datetime(2026, 6, 17),
    datetime(2026, 7, 29),
    datetime(2026, 9, 16),
    datetime(2026, 11, 4),
    datetime(2026, 12, 16),
]

ALL_FOMC_DATES = FOMC_DATES_2024 + FOMC_DATES_2025 + FOMC_DATES_2026


class EconomicCalendarCollector:
    """Collects and tracks economic calendar events with FRED data integration"""

    def __init__(self, fred_api_key: str = None):
        self.fred_api_key = fred_api_key or get_secret('FRED_API_KEY')
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 3600  # 1 hour cache
        self._fred_data_cache = {}
        self._fred_cache_time = None
        self._fred_cache_ttl = 1800  # 30 min cache for FRED data

    def _fetch_fred_series(self, series_id: str, observations: int = 15) -> Optional[List[Dict]]:
        """
        Fetch recent observations from FRED API

        Args:
            series_id: FRED series ID (e.g., 'CPIAUCSL')
            observations: Number of recent observations to fetch

        Returns:
            List of {date, value} dicts or None if error
        """
        if not self.fred_api_key:
            logger.debug("No FRED API key configured, skipping data fetch")
            return None

        # Check cache
        cache_key = f"{series_id}_{observations}"
        if self._fred_cache_time and (datetime.now() - self._fred_cache_time).seconds < self._fred_cache_ttl:
            if cache_key in self._fred_data_cache:
                return self._fred_data_cache[cache_key]

        try:
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': observations
            }

            response = requests.get(self.fred_base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            observations_data = data.get('observations', [])

            if not observations_data:
                return None

            # Parse and clean data
            result = []
            for obs in observations_data:
                if obs.get('value') and obs['value'] != '.':
                    result.append({
                        'date': obs['date'],
                        'value': float(obs['value'])
                    })

            # Cache results
            self._fred_data_cache[cache_key] = result
            self._fred_cache_time = datetime.now()

            return result

        except Exception as e:
            logger.debug(f"Error fetching FRED series {series_id}: {e}")
            return None

    def _get_indicator_data(self, series_id: str) -> Dict:
        """
        Get current, previous, and YoY change for an indicator

        Returns:
            Dict with 'current', 'previous', 'yoy_change' keys
        """
        data = self._fetch_fred_series(series_id, observations=15)

        if not data or len(data) < 2:
            return {'current': None, 'previous': None, 'yoy_change': None}

        current = data[0]['value']  # Most recent
        previous = data[1]['value']  # Prior reading

        # Calculate YoY change if we have 12+ months of data
        yoy_change = None
        if len(data) >= 13:
            year_ago = data[12]['value']
            if year_ago and year_ago != 0:
                yoy_change = ((current - year_ago) / year_ago) * 100

        return {
            'current': current,
            'previous': previous,
            'yoy_change': round(yoy_change, 2) if yoy_change else None,
            'current_date': data[0]['date'],
            'previous_date': data[1]['date']
        }

    def get_upcoming_events(self, days: int = 30) -> List[EconomicEvent]:
        """
        Get upcoming economic events

        Args:
            days: Number of days to look ahead

        Returns:
            List of EconomicEvent objects sorted by date
        """
        events = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = today + timedelta(days=days)

        # Add FOMC meetings
        events.extend(self._get_fomc_events(today, end_date))

        # Add CPI releases (typically 2nd week of month, 8:30 AM ET)
        events.extend(self._get_cpi_events(today, end_date))

        # Add NFP/Jobs Report (first Friday of month, 8:30 AM ET)
        events.extend(self._get_nfp_events(today, end_date))

        # Add GDP releases (quarterly, end of month)
        events.extend(self._get_gdp_events(today, end_date))

        # Add PCE (Fed's preferred inflation measure)
        events.extend(self._get_pce_events(today, end_date))

        # Add Retail Sales
        events.extend(self._get_retail_sales_events(today, end_date))

        # Add ISM Manufacturing
        events.extend(self._get_ism_events(today, end_date))

        # Sort by date
        events.sort(key=lambda x: x.date)

        return events

    def _get_fomc_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """Get FOMC meeting dates in range with Fed Funds rate data"""
        events = []
        fed_data = self._get_indicator_data(FRED_SERIES['FED_FUNDS'])

        for date in ALL_FOMC_DATES:
            if start <= date <= end:
                events.append(EconomicEvent(
                    name="FOMC Rate Decision",
                    date=date,
                    importance=EventImportance.HIGH,
                    category="Central Bank",
                    description="Federal Reserve interest rate decision and statement",
                    previous=fed_data.get('current'),  # Current rate before meeting
                    unit="%",
                    fred_series=FRED_SERIES['FED_FUNDS']
                ))
        return events

    def _get_cpi_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        CPI is typically released 2nd or 3rd week of month
        for the previous month's data. Includes actual YoY inflation from FRED.
        """
        events = []
        current = start.replace(day=1)

        # Fetch CPI data from FRED
        cpi_data = self._get_indicator_data(FRED_SERIES['CPI'])

        while current <= end:
            # CPI typically released around 10th-14th of month
            cpi_date = self._find_release_date(current.year, current.month, target_day=12)

            if start <= cpi_date <= end:
                events.append(EconomicEvent(
                    name="CPI (Inflation)",
                    date=cpi_date,
                    importance=EventImportance.HIGH,
                    category="Inflation",
                    description="Consumer Price Index - key inflation measure",
                    actual=cpi_data.get('current'),
                    previous=cpi_data.get('previous'),
                    yoy_change=cpi_data.get('yoy_change'),
                    unit="% YoY",
                    fred_series=FRED_SERIES['CPI']
                ))

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_nfp_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        Non-Farm Payrolls released first Friday of each month.
        Includes jobs added and unemployment rate from FRED.
        """
        events = []
        current = start.replace(day=1)

        # Fetch employment data from FRED
        nfp_data = self._get_indicator_data(FRED_SERIES['NFP'])
        unemp_data = self._get_indicator_data(FRED_SERIES['UNEMPLOYMENT'])

        # Calculate month-over-month change (jobs added)
        jobs_added = None
        if nfp_data.get('current') and nfp_data.get('previous'):
            jobs_added = nfp_data['current'] - nfp_data['previous']

        while current <= end:
            # Find first Friday of month
            nfp_date = self._find_first_friday(current.year, current.month)

            if start <= nfp_date <= end:
                events.append(EconomicEvent(
                    name="NFP (Jobs Report)",
                    date=nfp_date,
                    importance=EventImportance.HIGH,
                    category="Employment",
                    description=f"Non-Farm Payrolls. Unemployment: {unemp_data.get('current', 'N/A')}%",
                    actual=jobs_added,
                    previous=nfp_data.get('previous'),
                    unit="K jobs",
                    fred_series=FRED_SERIES['NFP']
                ))

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_gdp_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        GDP released ~end of month following quarter end
        Q1 (Jan-Mar) -> Late April
        Q2 (Apr-Jun) -> Late July
        Q3 (Jul-Sep) -> Late October
        Q4 (Oct-Dec) -> Late January
        """
        events = []
        gdp_months = [1, 4, 7, 10]  # Release months
        gdp_data = self._get_indicator_data(FRED_SERIES['GDP'])

        current = start.replace(day=1)
        while current <= end:
            if current.month in gdp_months:
                gdp_date = self._find_release_date(current.year, current.month, target_day=26)
                if start <= gdp_date <= end:
                    quarter = {1: 'Q4', 4: 'Q1', 7: 'Q2', 10: 'Q3'}[current.month]
                    events.append(EconomicEvent(
                        name=f"GDP ({quarter})",
                        date=gdp_date,
                        importance=EventImportance.HIGH,
                        category="Growth",
                        description=f"Gross Domestic Product - {quarter} economic growth",
                        actual=gdp_data.get('current'),
                        previous=gdp_data.get('previous'),
                        yoy_change=gdp_data.get('yoy_change'),
                        unit="% QoQ",
                        fred_series=FRED_SERIES['GDP']
                    ))

            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_pce_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """
        PCE (Personal Consumption Expenditures) - Fed's preferred inflation measure
        Released ~end of month for prior month. Includes actual YoY from FRED.
        """
        events = []
        current = start.replace(day=1)

        # Fetch PCE data from FRED
        pce_data = self._get_indicator_data(FRED_SERIES['PCE'])

        while current <= end:
            pce_date = self._find_release_date(current.year, current.month, target_day=28)
            if start <= pce_date <= end:
                events.append(EconomicEvent(
                    name="PCE Inflation",
                    date=pce_date,
                    importance=EventImportance.HIGH,
                    category="Inflation",
                    description="Fed's preferred inflation gauge",
                    actual=pce_data.get('current'),
                    previous=pce_data.get('previous'),
                    yoy_change=pce_data.get('yoy_change'),
                    unit="% YoY",
                    fred_series=FRED_SERIES['PCE']
                ))

            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_retail_sales_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """Retail Sales - mid-month. Includes YoY change from FRED."""
        events = []
        current = start.replace(day=1)

        # Fetch retail sales data
        retail_data = self._get_indicator_data(FRED_SERIES['RETAIL'])

        while current <= end:
            rs_date = self._find_release_date(current.year, current.month, target_day=15)
            if start <= rs_date <= end:
                events.append(EconomicEvent(
                    name="Retail Sales",
                    date=rs_date,
                    importance=EventImportance.MEDIUM,
                    category="Consumer",
                    description="Monthly retail sales data",
                    actual=retail_data.get('current'),
                    previous=retail_data.get('previous'),
                    yoy_change=retail_data.get('yoy_change'),
                    unit="% MoM",
                    fred_series=FRED_SERIES['RETAIL']
                ))

            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_ism_events(self, start: datetime, end: datetime) -> List[EconomicEvent]:
        """ISM Manufacturing - first business day of month"""
        events = []
        current = start.replace(day=1)

        while current <= end:
            ism_date = self._find_first_business_day(current.year, current.month)
            if start <= ism_date <= end:
                events.append(EconomicEvent(
                    name="ISM Manufacturing",
                    date=ism_date,
                    importance=EventImportance.MEDIUM,
                    category="Manufacturing",
                    description="Manufacturing sector health indicator"
                ))

            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _find_first_friday(self, year: int, month: int) -> datetime:
        """Find the first Friday of a given month"""
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        return first_day + timedelta(days=days_until_friday)

    def _find_first_business_day(self, year: int, month: int) -> datetime:
        """Find first business day (Mon-Fri) of month"""
        first_day = datetime(year, month, 1)
        while first_day.weekday() >= 5:  # Saturday=5, Sunday=6
            first_day += timedelta(days=1)
        return first_day

    def _find_release_date(self, year: int, month: int, target_day: int) -> datetime:
        """Find a release date, adjusting for weekends"""
        try:
            date = datetime(year, month, min(target_day, 28))
        except ValueError:
            date = datetime(year, month, 28)

        # Move to next business day if weekend
        while date.weekday() >= 5:
            date += timedelta(days=1)
        return date

    def get_next_major_event(self) -> Optional[EconomicEvent]:
        """Get the next major (HIGH importance) economic event"""
        events = self.get_upcoming_events(days=60)
        for event in events:
            if event.importance == EventImportance.HIGH and event.date > datetime.now():
                return event
        return None

    def get_events_this_week(self) -> List[EconomicEvent]:
        """Get economic events for the current week"""
        today = datetime.now()
        # Find Monday of this week
        monday = today - timedelta(days=today.weekday())
        friday = monday + timedelta(days=4)

        events = self.get_upcoming_events(days=14)
        return [e for e in events if monday <= e.date <= friday + timedelta(days=1)]

    def get_countdown_to_next_event(self) -> Optional[Dict]:
        """
        Get countdown to next major event

        Returns:
            Dict with event info and time until event
        """
        event = self.get_next_major_event()
        if not event:
            return None

        now = datetime.now()
        delta = event.date - now

        return {
            'event': event.name,
            'date': event.date,
            'category': event.category,
            'description': event.description,
            'days': delta.days,
            'hours': delta.seconds // 3600,
            'total_hours': delta.total_seconds() / 3600,
            'is_today': delta.days == 0,
            'is_tomorrow': delta.days == 1,
        }

    def get_calendar_summary(self) -> Dict:
        """
        Get summary of upcoming calendar

        Returns:
            Dict with event counts and next events by category
        """
        events = self.get_upcoming_events(days=30)

        summary = {
            'total_events': len(events),
            'high_importance': len([e for e in events if e.importance == EventImportance.HIGH]),
            'this_week': len(self.get_events_this_week()),
            'next_fomc': None,
            'next_cpi': None,
            'next_nfp': None,
            'next_event': None,
        }

        for event in events:
            if event.date > datetime.now():
                if summary['next_event'] is None:
                    summary['next_event'] = {
                        'name': event.name,
                        'date': event.date,
                        'days_away': (event.date - datetime.now()).days
                    }
                if 'FOMC' in event.name and summary['next_fomc'] is None:
                    summary['next_fomc'] = event.date
                if 'CPI' in event.name and summary['next_cpi'] is None:
                    summary['next_cpi'] = event.date
                if 'NFP' in event.name and summary['next_nfp'] is None:
                    summary['next_nfp'] = event.date

        return summary

    def get_latest_economic_data(self) -> Dict:
        """
        Get latest values for key economic indicators from FRED.

        Returns:
            Dict with current values for CPI, NFP, GDP, PCE, Unemployment
        """
        indicators = {}

        # CPI - Inflation
        cpi = self._get_indicator_data(FRED_SERIES['CPI'])
        if cpi.get('yoy_change') is not None:
            indicators['cpi'] = {
                'name': 'CPI Inflation',
                'value': cpi['yoy_change'],
                'unit': '% YoY',
                'date': cpi.get('current_date', 'N/A'),
                'previous': cpi.get('previous'),
                'trend': 'rising' if cpi['yoy_change'] > 3.0 else 'falling' if cpi['yoy_change'] < 2.0 else 'stable'
            }

        # Unemployment
        unemp = self._get_indicator_data(FRED_SERIES['UNEMPLOYMENT'])
        if unemp.get('current') is not None:
            indicators['unemployment'] = {
                'name': 'Unemployment Rate',
                'value': unemp['current'],
                'unit': '%',
                'date': unemp.get('current_date', 'N/A'),
                'previous': unemp.get('previous')
            }

        # Fed Funds Rate
        fed = self._get_indicator_data(FRED_SERIES['FED_FUNDS'])
        if fed.get('current') is not None:
            indicators['fed_funds'] = {
                'name': 'Fed Funds Rate',
                'value': fed['current'],
                'unit': '%',
                'date': fed.get('current_date', 'N/A')
            }

        # PCE Inflation
        pce = self._get_indicator_data(FRED_SERIES['PCE'])
        if pce.get('yoy_change') is not None:
            indicators['pce'] = {
                'name': 'PCE Inflation (Fed Target)',
                'value': pce['yoy_change'],
                'unit': '% YoY',
                'target': 2.0,  # Fed's target
                'above_target': pce['yoy_change'] > 2.0
            }

        # NFP Jobs
        nfp = self._get_indicator_data(FRED_SERIES['NFP'])
        if nfp.get('current') and nfp.get('previous'):
            jobs_added = nfp['current'] - nfp['previous']
            indicators['jobs'] = {
                'name': 'Jobs Added (Last Month)',
                'value': jobs_added,
                'unit': 'K',
                'date': nfp.get('current_date', 'N/A')
            }

        return {
            'indicators': indicators,
            'has_fred_api': self.fred_api_key is not None,
            'timestamp': datetime.now().isoformat()
        }
