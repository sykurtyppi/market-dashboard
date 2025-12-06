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
        "liquidity": 24,     # Daily FRED updates (RRP, TGA, SOFR)
        "vrp": 24,          # Daily VRP calculation
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
    
    def check_data_source(self, source_name: str, column_name: str) -> DataSourceHealth:
        """
        Check health of a specific data source
        
        Args:
            source_name: Display name for the source
            column_name: Database column name to check
        
        Returns:
            DataSourceHealth object
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get most recent non-null value
                query = f"""
                    SELECT date, {column_name}
                    FROM daily_snapshots
                    WHERE {column_name} IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 1
                """
                
                cursor.execute(query)
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
    
    def check_indicator(self, indicator_name: str, display_name: str = None) -> DataSourceHealth:
        """
        Check health of an indicator in the indicators table
        
        Args:
            indicator_name: Name in indicators table
            display_name: Display name (uses indicator_name if None)
        
        Returns:
            DataSourceHealth object
        """
        if display_name is None:
            display_name = indicator_name
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT date, value
                    FROM indicators
                    WHERE indicator_name = ?
                    ORDER BY date DESC
                    LIMIT 1
                """
                
                cursor.execute(query, (indicator_name,))
                result = cursor.fetchone()
                
                if not result:
                    return DataSourceHealth(
                        name=display_name,
                        status=HealthStatus.UNKNOWN,
                        last_update=None,
                        message="No data available"
                    )
                
                last_date_str, value = result
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                
                age = datetime.now() - last_date
                age_hours = age.total_seconds() / 3600
                
                threshold = 72  # 3 days for indicators
                
                if age_hours < threshold:
                    status = HealthStatus.HEALTHY
                    message = f"Current (last: {last_date.strftime('%Y-%m-%d')})"
                elif age_hours < threshold * 2:
                    status = HealthStatus.STALE
                    message = f"Stale ({age_hours:.1f}h old)"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Very stale ({age_hours:.1f}h old)"
                
                return DataSourceHealth(
                    name=display_name,
                    status=status,
                    last_update=last_date,
                    message=message,
                    age_hours=age_hours
                )
                
        except Exception as e:
            return DataSourceHealth(
                name=display_name,
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
        
        # Liquidity indicators (NEW!)
        checks["liquidity_rrp"] = self.check_indicator("liquidity_rrp", "Fed RRP")
        checks["liquidity_tga"] = self.check_indicator("liquidity_tga", "TGA Balance")
        checks["liquidity_net"] = self.check_indicator("liquidity_net", "Net Liquidity")
        
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
            HealthStatus.DEGRADED: "🔶",
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


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA HEALTH CHECK SYSTEM - TEST")
    print("="*80)
    
    health_checker = HealthCheckSystem()
    
    print("\n Running comprehensive health check...")
    summary = health_checker.get_health_summary()
    
    print(f"\n OVERALL STATUS: {summary['overall_status'].upper()}")
    print(f"   Checked: {summary['total_sources']} sources")
    
    print(f"\n STATUS BREAKDOWN:")
    for status, count in summary['summary'].items():
        if count > 0:
            print(f"   {status.title()}: {count}")
    
    print(f"\n INDIVIDUAL SOURCES:")
    for name, check_data in summary['sources'].items():
        emoji = health_checker.get_status_emoji(HealthStatus(check_data['status']))
        print(f"   {emoji} {check_data['name']}: {check_data['message']}")
    
    print("\n" + "="*80)
    print("✓ HEALTH CHECK COMPLETE")
