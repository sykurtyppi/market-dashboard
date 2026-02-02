"""
Data Status Tracking System

Provides unified status tracking for all data collectors and database operations.
Helps distinguish between:
- Fresh data (OK)
- Stale data (old but valid)
- Estimated data (calculated fallbacks)
- Unavailable data (API offline)
- Error conditions (failures)
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Status of data freshness and reliability"""
    OK = "ok"                    # Fresh, real data
    STALE = "stale"              # Data older than threshold but valid
    ESTIMATED = "estimated"       # Calculated/fallback values
    UNAVAILABLE = "unavailable"   # API offline or no data exists
    ERROR = "error"              # Fetch/processing error occurred
    PARTIAL = "partial"          # Some fields missing or estimated


@dataclass
class DataResult:
    """
    Wrapper for data with status metadata.

    Use this instead of returning raw DataFrames or dicts from collectors.
    """
    data: Any                           # The actual data (DataFrame, dict, etc.)
    status: DataStatus                  # Overall status
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""                    # Data source identifier (e.g., "FRED", "Yahoo", "CBOE")
    message: str = ""                   # Human-readable status message
    age_hours: float = 0.0              # How old is this data
    estimated_fields: List[str] = field(default_factory=list)  # Which fields are estimated
    errors: List[str] = field(default_factory=list)            # Any errors encountered

    @property
    def is_fresh(self) -> bool:
        """Data is fresh (less than 4 hours old during market hours)"""
        return self.status == DataStatus.OK and self.age_hours < 4

    @property
    def is_usable(self) -> bool:
        """Data can be displayed (OK, STALE, ESTIMATED, or PARTIAL)"""
        return self.status in (DataStatus.OK, DataStatus.STALE, DataStatus.ESTIMATED, DataStatus.PARTIAL)

    @property
    def has_warnings(self) -> bool:
        """Data has warnings that should be shown to user"""
        return self.status in (DataStatus.STALE, DataStatus.ESTIMATED, DataStatus.PARTIAL) or len(self.estimated_fields) > 0

    @property
    def status_emoji(self) -> str:
        """Emoji indicator for status"""
        return {
            DataStatus.OK: "✅",
            DataStatus.STALE: "⚠️",
            DataStatus.ESTIMATED: "📊",
            DataStatus.UNAVAILABLE: "❌",
            DataStatus.ERROR: "🚨",
            DataStatus.PARTIAL: "⚡",
        }.get(self.status, "❓")

    @property
    def status_color(self) -> str:
        """Color for UI display"""
        return {
            DataStatus.OK: "#4CAF50",       # Green
            DataStatus.STALE: "#FF9800",    # Orange
            DataStatus.ESTIMATED: "#2196F3", # Blue
            DataStatus.UNAVAILABLE: "#9E9E9E", # Gray
            DataStatus.ERROR: "#F44336",    # Red
            DataStatus.PARTIAL: "#FF9800",  # Orange
        }.get(self.status, "#9E9E9E")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "message": self.message,
            "age_hours": self.age_hours,
            "estimated_fields": self.estimated_fields,
            "errors": self.errors,
            "is_fresh": self.is_fresh,
            "is_usable": self.is_usable,
        }


def calculate_data_age(data_timestamp: datetime) -> float:
    """Calculate age of data in hours"""
    if data_timestamp is None:
        return float('inf')

    now = datetime.now()

    # Handle timezone-naive vs timezone-aware
    if data_timestamp.tzinfo is not None and now.tzinfo is None:
        data_timestamp = data_timestamp.replace(tzinfo=None)

    age = now - data_timestamp
    return age.total_seconds() / 3600  # Convert to hours


def get_staleness_status(age_hours: float, thresholds: Dict[str, float] = None) -> DataStatus:
    """
    Determine status based on data age.

    Default thresholds:
    - OK: < 4 hours (within trading day)
    - STALE: 4-24 hours (overnight acceptable)
    - UNAVAILABLE: > 24 hours (too old)
    """
    if thresholds is None:
        thresholds = {
            "fresh": 4,      # Hours
            "stale": 24,     # Hours
        }

    if age_hours < thresholds["fresh"]:
        return DataStatus.OK
    elif age_hours < thresholds["stale"]:
        return DataStatus.STALE
    else:
        return DataStatus.UNAVAILABLE


def format_age_string(age_hours: float) -> str:
    """Format age as human-readable string"""
    if age_hours < 0.017:  # Less than 1 minute
        return "just now"
    elif age_hours < 1:
        minutes = int(age_hours * 60)
        return f"{minutes}m ago"
    elif age_hours < 24:
        hours = int(age_hours)
        return f"{hours}h ago"
    elif age_hours < 168:  # Less than 1 week
        days = int(age_hours / 24)
        return f"{days}d ago"
    else:
        weeks = int(age_hours / 168)
        return f"{weeks}w ago"


class DataStatusTracker:
    """
    Track data status across multiple sources.

    Usage:
        tracker = DataStatusTracker()
        tracker.update("FRED", DataStatus.OK, age_hours=0.5)
        tracker.update("CBOE", DataStatus.ESTIMATED, estimated_fields=["vix3m"])

        summary = tracker.get_summary()
        # Returns: {"overall": "PARTIAL", "sources": {...}, "warnings": [...]}
    """

    def __init__(self):
        self.sources: Dict[str, DataResult] = {}

    def update(
        self,
        source: str,
        status: DataStatus,
        data: Any = None,
        message: str = "",
        age_hours: float = 0.0,
        estimated_fields: List[str] = None,
        errors: List[str] = None
    ):
        """Update status for a data source"""
        self.sources[source] = DataResult(
            data=data,
            status=status,
            source=source,
            message=message,
            age_hours=age_hours,
            estimated_fields=estimated_fields or [],
            errors=errors or []
        )

    def get_source(self, source: str) -> Optional[DataResult]:
        """Get status for a specific source"""
        return self.sources.get(source)

    def get_overall_status(self) -> DataStatus:
        """Get overall status across all sources"""
        if not self.sources:
            return DataStatus.UNAVAILABLE

        statuses = [r.status for r in self.sources.values()]

        # If any errors, overall is ERROR
        if DataStatus.ERROR in statuses:
            return DataStatus.ERROR

        # If any unavailable, overall is PARTIAL
        if DataStatus.UNAVAILABLE in statuses:
            return DataStatus.PARTIAL

        # If any stale or estimated, overall is PARTIAL
        if DataStatus.STALE in statuses or DataStatus.ESTIMATED in statuses:
            return DataStatus.PARTIAL

        # All OK
        return DataStatus.OK

    def get_warnings(self) -> List[str]:
        """Get list of warning messages"""
        warnings = []

        for source, result in self.sources.items():
            if result.status == DataStatus.STALE:
                warnings.append(f"{source}: Data is {format_age_string(result.age_hours)} old")
            elif result.status == DataStatus.ESTIMATED:
                fields = ", ".join(result.estimated_fields) if result.estimated_fields else "some fields"
                warnings.append(f"{source}: {fields} estimated")
            elif result.status == DataStatus.ERROR:
                error_msg = result.errors[0] if result.errors else "Unknown error"
                warnings.append(f"{source}: {error_msg}")
            elif result.status == DataStatus.UNAVAILABLE:
                warnings.append(f"{source}: Data unavailable")

        return warnings

    def get_summary(self) -> Dict[str, Any]:
        """Get complete status summary"""
        return {
            "overall_status": self.get_overall_status().value,
            "overall_emoji": self.sources and list(self.sources.values())[0].status_emoji or "❓",
            "sources": {
                name: {
                    "status": result.status.value,
                    "emoji": result.status_emoji,
                    "age": format_age_string(result.age_hours),
                    "message": result.message,
                    "estimated_fields": result.estimated_fields,
                }
                for name, result in self.sources.items()
            },
            "warnings": self.get_warnings(),
            "has_warnings": any(r.has_warnings for r in self.sources.values()),
            "all_fresh": all(r.is_fresh for r in self.sources.values()),
        }


# Convenience functions for common patterns

def ok_result(data: Any, source: str, message: str = "Data fetched successfully") -> DataResult:
    """Create an OK result"""
    return DataResult(
        data=data,
        status=DataStatus.OK,
        source=source,
        message=message,
        age_hours=0.0
    )


def stale_result(data: Any, source: str, age_hours: float, message: str = "") -> DataResult:
    """Create a STALE result"""
    if not message:
        message = f"Data is {format_age_string(age_hours)} old"
    return DataResult(
        data=data,
        status=DataStatus.STALE,
        source=source,
        message=message,
        age_hours=age_hours
    )


def estimated_result(
    data: Any,
    source: str,
    estimated_fields: List[str],
    message: str = ""
) -> DataResult:
    """Create an ESTIMATED result"""
    if not message:
        message = f"Fields estimated: {', '.join(estimated_fields)}"
    return DataResult(
        data=data,
        status=DataStatus.ESTIMATED,
        source=source,
        message=message,
        estimated_fields=estimated_fields
    )


def error_result(source: str, error: str, data: Any = None) -> DataResult:
    """Create an ERROR result"""
    logger.error(f"[{source}] {error}")
    return DataResult(
        data=data,
        status=DataStatus.ERROR,
        source=source,
        message=error,
        errors=[error]
    )


def unavailable_result(source: str, reason: str = "Data not available") -> DataResult:
    """Create an UNAVAILABLE result"""
    return DataResult(
        data=None,
        status=DataStatus.UNAVAILABLE,
        source=source,
        message=reason
    )
