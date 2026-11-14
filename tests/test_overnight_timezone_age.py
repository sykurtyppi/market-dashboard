from datetime import datetime, timedelta, timezone

from utils import data_status


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        instant = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)


def test_aware_age_preserves_instant(monkeypatch):
    monkeypatch.setattr(data_status, "datetime", FixedClock)
    timestamp = datetime(2026, 11, 14, 13, tzinfo=timezone(timedelta(hours=2)))
    assert data_status.calculate_data_age(timestamp) == 1.0


def test_naive_age_retains_local_wall_time(monkeypatch):
    monkeypatch.setattr(data_status, "datetime", FixedClock)
    assert data_status.calculate_data_age(datetime(2026, 11, 14, 11)) == 1.0
