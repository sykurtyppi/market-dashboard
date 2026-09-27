"""
Data Health Check System
Monitors status and freshness of all data sources
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    STALE = "stale"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class DataSourceHealth:
    """Health status for a single data source"""
    name: str
    status: HealthStatus
    last_update: Optional[datetime]
    message: str
    age_hours: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "message": self.message,
            "age_hours": round(self.age_hours, 1) if self.age_hours else None
        }


class HealthCheckSystem:
    """Monitors health of all data sources"""
    
    # Freshness thresholds (in hours)
    FRESHNESS_THRESHOLDS = {
        "vix": 24,           # Daily during market hours
        "credit_spread": 24, # Daily FRED updates
        "fear_greed": 24,    # Daily updates
        "treasury": 24,      # Daily FRED updates
        "breadth": 24,       # Daily calculations
        "vrp": 24,          # Daily VRP calculation
        # Liquidity series carry publication lag on top of the weekend gap, so
        # a 24h bar would flag them stale on every normal day. RRP posts each
        # business day (Monday sees Friday's: ~72h); TGA posts T+2 (a weekend
        # stretches that to ~96h); net liquidity needs both. Sized so a normal
        # lag is "healthy" and only a stalled pipeline trips "stale".
        "fed_rrp": 96,
        "tga_balance": 120,
        "net_liquidity": 120,
    }
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
    
    def check_database_connection(self) -> DataSourceHealth:
        """Check if database is accessible"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                
            return DataSourceHealth(
                name="Database",
                status=HealthStatus.HEALTHY,
                last_update=datetime.now(),
                message="Database connection OK"
            )
            
        except Exception as e:
            return DataSourceHealth(
                name="Database",
                status=HealthStatus.DOWN,
                last_update=None,
                message=f"Database connection failed: {str(e)}"
            )
    
    # Allowed table/column pairs for SQL queries (prevent SQL injection).
    # Liquidity lives in liquidity_history, not daily_snapshots — the snapshot
    # table has no rrp/tga/net_liquidity columns.
    ALLOWED_COLUMNS = {
        'credit_spread_hy', 'credit_spread_ig', 'treasury_10y', 'fed_funds',
        'vix_spot', 'vix9d', 'vvix', 'vvix_signal', 'skew', 'vrp',
        'vix_contango', 'put_call_ratio', 'fear_greed_score', 'market_breadth',
        'left_signal', 'move',
    }
    ALLOWED_TABLES = {
        'daily_snapshots': frozenset(ALLOWED_COLUMNS),
        'liquidity_history': frozenset({'rrp_on', 'tga', 'net_liquidity', 'sofr', 'fed_balance_sheet'}),
    }

    def check_data_source(
        self, source_name: str, column_name: str, table: str = "daily_snapshots"
    ) -> DataSourceHealth:
        """
        Check health of a specific data source

        Args:
            source_name: Display name for the source
            column_name: Database column name to check
            table: Table holding the column (must be in ALLOWED_TABLES)

        Returns:
            DataSourceHealth object
        """
        # SECURITY: Validate table and column names to prevent SQL injection
        allowed = self.ALLOWED_TABLES.get(table)
        if allowed is None:
            return DataSourceHealth(
                name=source_name,
                status=HealthStatus.UNKNOWN,
                last_update=None,
                message=f"Invalid table name: {table}"
            )
        if column_name not in allowed:
            return DataSourceHealth(
                name=source_name,
                status=HealthStatus.UNKNOWN,
                last_update=None,
                message=f"Invalid column name: {column_name}"
            )

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Most recent NON-NULL value. Rows can exist with the column
                # still empty (liquidity_history carries staggered nulls while
                # each series catches up), so the latest row is not the latest
                # observation.
                query = f"""
                    SELECT date, {column_name}
                    FROM {table}
                    WHERE {column_name} IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 1
                """

                try:
                    cursor.execute(query)
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        # Fresh or pre-migration database. Report it the way an
                        # empty table is reported rather than dragging overall
                        # health to DOWN over a table no refresh has created yet.
                        return DataSourceHealth(
                            name=source_name,
                            status=HealthStatus.UNKNOWN,
                            last_update=None,
                            message=f"Table {table} not present (no refresh has run yet)"
                        )
                    raise
                result = cursor.fetchone()
                
                if not result:
                    return DataSourceHealth(
                        name=source_name,
                        status=HealthStatus.UNKNOWN,
                        last_update=None,
                        message="No data available"
                    )
                
                last_date_str, value = result
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                
                # Calculate age
                age = datetime.now() - last_date
                age_hours = age.total_seconds() / 3600
                
                # Determine status based on freshness
                threshold = self.FRESHNESS_THRESHOLDS.get(
                    source_name.lower().replace(" ", "_"),
                    24
                )
                
                if age_hours < threshold:
                    status = HealthStatus.HEALTHY
                    message = f"Data current (last: {last_date.strftime('%Y-%m-%d')})"
                elif age_hours < threshold * 2:
                    status = HealthStatus.STALE
                    message = f"Data stale ({age_hours:.1f}h old)"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Data very stale ({age_hours:.1f}h old)"
                
                return DataSourceHealth(
                    name=source_name,
                    status=status,
                    last_update=last_date,
                    message=message,
                    age_hours=age_hours
                )
                
        except Exception as e:
            return DataSourceHealth(
                name=source_name,
                status=HealthStatus.DOWN,
                last_update=None,
                message=f"Check failed: {str(e)}"
            )
    
    def get_all_health_checks(self) -> Dict[str, DataSourceHealth]:
        """
        Run health checks on all data sources
        
        Returns:
            Dict mapping source name to health status
        """
        checks = {}
        
        # Database connection
        checks["database"] = self.check_database_connection()
        
        # Only proceed if database is accessible
        if checks["database"].status == HealthStatus.DOWN:
            return checks
        
        # Daily snapshot sources
        checks["vix"] = self.check_data_source("VIX", "vix_spot")
        checks["credit_hy"] = self.check_data_source("Credit Spread (HY)", "credit_spread_hy")
        checks["treasury_10y"] = self.check_data_source("10Y Treasury", "treasury_10y")
        checks["fear_greed"] = self.check_data_source("Fear & Greed", "fear_greed_score")
        checks["put_call"] = self.check_data_source("Put/Call Ratio", "put_call_ratio")
        
        # Liquidity series are written to liquidity_history by the refresh
        # (scheduler.daily_update), never to the indicators table — checking
        # indicators for them reported "No data available" on a healthy system.
        checks["liquidity_rrp"] = self.check_data_source("Fed RRP", "rrp_on", table="liquidity_history")
        checks["liquidity_tga"] = self.check_data_source("TGA Balance", "tga", table="liquidity_history")
        checks["liquidity_net"] = self.check_data_source("Net Liquidity", "net_liquidity", table="liquidity_history")
        
        return checks
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health
        
        Returns:
            Worst status across all sources
        """
        all_checks = self.get_all_health_checks()
        
        statuses = [check.status for check in all_checks.values()]
        
        # Return worst status
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.STALE in statuses:
            return HealthStatus.STALE
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict:
        """
        Get summary of system health
        
        Returns:
            Dict with overall status and individual source statuses
        """
        all_checks = self.get_all_health_checks()
        overall = self.get_overall_health()
        
        # Count by status
        status_counts = {
            "healthy": 0,
            "stale": 0,
            "degraded": 0,
            "down": 0,
            "unknown": 0
        }
        
        for check in all_checks.values():
            status_counts[check.status.value] += 1
        
        return {
            "overall_status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "sources": {name: check.to_dict() for name, check in all_checks.items()},
            "summary": status_counts,
            "total_sources": len(all_checks)
        }
    
    def get_status_emoji(self, status: HealthStatus) -> str:
        """Get emoji for status visualization"""
        emoji_map = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.STALE: "⚠️",
            HealthStatus.DEGRADED: "Degraded",
            HealthStatus.DOWN: "❌",
            HealthStatus.UNKNOWN: "❓"
        }
        return emoji_map.get(status, "❓")
    
    def get_status_color(self, status: HealthStatus) -> str:
        """Get color code for status visualization"""
        color_map = {
            HealthStatus.HEALTHY: "#4CAF50",    # Green
            HealthStatus.STALE: "#FFC107",      # Amber
            HealthStatus.DEGRADED: "#FF9800",   # Orange
            HealthStatus.DOWN: "#F44336",       # Red
            HealthStatus.UNKNOWN: "#9E9E9E"     # Grey
        }
        return color_map.get(status, "#9E9E9E")
