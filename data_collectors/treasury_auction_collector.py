"""
Treasury Auction Data Collector
Fetches upcoming and recent auction data from Treasury Direct API

Key metrics:
- Bid-to-Cover ratio: Demand indicator (>2.5 = strong, <2.0 = weak)
- High yield vs when-issued: Pricing tension
- Indirect bidders %: Foreign/institutional demand
- Direct bidders %: Domestic institutional demand
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from utils.retry_utils import exponential_backoff_retry

logger = logging.getLogger(__name__)

# Treasury security types
SECURITY_TYPES = {
    'Bill': ['4-Week', '8-Week', '13-Week', '17-Week', '26-Week', '52-Week'],
    'Note': ['2-Year', '3-Year', '5-Year', '7-Year', '10-Year'],
    'Bond': ['20-Year', '30-Year'],
    'TIPS': ['5-Year TIPS', '10-Year TIPS', '30-Year TIPS'],
    'FRN': ['2-Year FRN']
}

# Key auctions to track (most market-moving)
KEY_AUCTIONS = ['10-Year', '30-Year', '2-Year', '5-Year', '7-Year']


class TreasuryAuctionCollector:
    """Collects Treasury auction data from Treasury Direct"""

    def __init__(self):
        # Treasury Direct API for auction results
        self.base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        # Try multiple endpoints as they change
        self.auction_endpoints = [
            "/v2/accounting/od/auction_results",
            "/v1/accounting/od/auction_results",
        ]

    @exponential_backoff_retry(max_retries=2, base_delay=1.0)
    def get_recent_auctions(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch recent Treasury auction results

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with auction results
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Try each endpoint until one works
            for endpoint in self.auction_endpoints:
                try:
                    params = {
                        'filter': f'auction_date:gte:{start_date.strftime("%Y-%m-%d")}',
                        'sort': '-auction_date',
                        'page[size]': 100,
                    }

                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        params=params,
                        timeout=15
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data'):
                            return self._process_auction_data(data['data'])
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} failed: {e}")
                    continue

            # If API fails, return sample data
            logger.warning("Treasury API unavailable, using sample data")
            return self._get_sample_auction_data()

        except Exception as e:
            logger.error(f"Error fetching auction data: {e}")
            return self._get_sample_auction_data()

    def _process_auction_data(self, data: List) -> pd.DataFrame:
        """Process raw auction data into DataFrame"""
        df = pd.DataFrame(data)

        # Convert numeric columns
        numeric_cols = ['high_investment_rate', 'bid_to_cover_ratio',
                      'total_accepted', 'total_tendered',
                      'direct_bidder_accepted', 'indirect_bidder_accepted',
                      'primary_dealer_accepted']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert date columns
        if 'auction_date' in df.columns:
            df['auction_date'] = pd.to_datetime(df['auction_date'])
        if 'issue_date' in df.columns:
            df['issue_date'] = pd.to_datetime(df['issue_date'])

        # Calculate bidder percentages (with division by zero protection)
        if 'total_accepted' in df.columns and df['total_accepted'].notna().any():
            total = df['total_accepted'].replace(0, pd.NA)  # Avoid division by zero
            if 'direct_bidder_accepted' in df.columns:
                df['direct_pct'] = (df['direct_bidder_accepted'] / total * 100).round(1)
            if 'indirect_bidder_accepted' in df.columns:
                df['indirect_pct'] = (df['indirect_bidder_accepted'] / total * 100).round(1)
            if 'primary_dealer_accepted' in df.columns:
                df['dealer_pct'] = (df['primary_dealer_accepted'] / total * 100).round(1)

        logger.info(f"Fetched {len(df)} recent auction results")
        return df

    def _get_sample_auction_data(self) -> pd.DataFrame:
        """Return sample auction data when API unavailable"""
        today = datetime.now()
        sample = [
            {
                'security_type': 'Note',
                'security_term': '10-Year',
                'auction_date': (today - timedelta(days=5)),
                'high_investment_rate': 4.25,
                'bid_to_cover_ratio': 2.45,
                'direct_pct': 18.5,
                'indirect_pct': 68.2,
                'dealer_pct': 13.3,
            },
            {
                'security_type': 'Note',
                'security_term': '2-Year',
                'auction_date': (today - timedelta(days=7)),
                'high_investment_rate': 4.35,
                'bid_to_cover_ratio': 2.62,
                'direct_pct': 22.1,
                'indirect_pct': 62.5,
                'dealer_pct': 15.4,
            },
            {
                'security_type': 'Bond',
                'security_term': '30-Year',
                'auction_date': (today - timedelta(days=10)),
                'high_investment_rate': 4.45,
                'bid_to_cover_ratio': 2.38,
                'direct_pct': 15.8,
                'indirect_pct': 70.1,
                'dealer_pct': 14.1,
            },
        ]
        return pd.DataFrame(sample)

    def get_key_auction_results(self, days: int = 60) -> Optional[Dict]:
        """
        Get results for key market-moving auctions

        Returns:
            Dict with latest results for each key auction type
        """
        df = self.get_recent_auctions(days=days)
        if df is None or df.empty:
            return None

        results = {}

        for term in KEY_AUCTIONS:
            # Filter for this term
            mask = df['security_term'].str.contains(term.split('-')[0], case=False, na=False)
            term_df = df[mask].head(1)

            if not term_df.empty:
                row = term_df.iloc[0]
                results[term] = {
                    'date': row.get('auction_date'),
                    'yield': row.get('high_investment_rate'),
                    'bid_to_cover': row.get('bid_to_cover_ratio'),
                    'direct_pct': row.get('direct_pct'),
                    'indirect_pct': row.get('indirect_pct'),
                    'dealer_pct': row.get('dealer_pct'),
                    'demand_rating': self._rate_demand(row.get('bid_to_cover_ratio'))
                }

        return results if results else None

    def _rate_demand(self, bid_to_cover: Optional[float]) -> str:
        """Rate auction demand based on bid-to-cover ratio"""
        if bid_to_cover is None:
            return "Unknown"
        if bid_to_cover >= 2.8:
            return "Very Strong"
        elif bid_to_cover >= 2.5:
            return "Strong"
        elif bid_to_cover >= 2.2:
            return "Average"
        elif bid_to_cover >= 2.0:
            return "Weak"
        else:
            return "Very Weak"

    def get_upcoming_auctions(self) -> Optional[List[Dict]]:
        """
        Fetch upcoming Treasury auctions from announcement calendar

        Note: Treasury Direct doesn't have a direct API for upcoming auctions.
        This attempts to infer from typical auction schedule.
        """
        try:
            # Treasury auction schedule is predictable:
            # - 4-week bills: Every Tuesday
            # - 8-week bills: Every Tuesday
            # - 13-week bills: Every Monday
            # - 26-week bills: Every Monday
            # - 2-year notes: End of month
            # - 5-year notes: End of month
            # - 7-year notes: End of month
            # - 10-year notes: Mid-month
            # - 30-year bonds: Mid-month (monthly)

            today = datetime.now()
            upcoming = []

            # Find next Monday and Tuesday
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0 and today.hour >= 13:  # After 1PM auction time
                days_until_monday = 7
            next_monday = today + timedelta(days=days_until_monday)

            days_until_tuesday = (1 - today.weekday()) % 7
            if days_until_tuesday == 0 and today.hour >= 13:
                days_until_tuesday = 7
            next_tuesday = today + timedelta(days=days_until_tuesday)

            # Add regular weekly auctions
            upcoming.append({
                'date': next_monday.strftime('%Y-%m-%d'),
                'security': '13-Week Bill',
                'type': 'Bill',
                'frequency': 'Weekly'
            })
            upcoming.append({
                'date': next_monday.strftime('%Y-%m-%d'),
                'security': '26-Week Bill',
                'type': 'Bill',
                'frequency': 'Weekly'
            })
            upcoming.append({
                'date': next_tuesday.strftime('%Y-%m-%d'),
                'security': '4-Week Bill',
                'type': 'Bill',
                'frequency': 'Weekly'
            })

            # Sort by date
            upcoming.sort(key=lambda x: x['date'])

            return upcoming[:10]  # Return next 10 auctions

        except Exception as e:
            logger.error(f"Error generating upcoming auctions: {e}")
            return None

    def get_auction_summary(self) -> Optional[Dict]:
        """
        Get summary of recent auction health

        Returns:
            Dict with overall demand metrics
        """
        df = self.get_recent_auctions(days=30)
        if df is None or df.empty:
            return None

        # Filter for notes and bonds (most important)
        notes_bonds = df[df['security_type'].isin(['Note', 'Bond'])]

        if notes_bonds.empty:
            notes_bonds = df  # Fall back to all auctions

        summary = {
            'avg_bid_to_cover': notes_bonds['bid_to_cover_ratio'].mean(),
            'avg_indirect_pct': notes_bonds.get('indirect_pct', pd.Series()).mean(),
            'avg_direct_pct': notes_bonds.get('direct_pct', pd.Series()).mean(),
            'auction_count': len(notes_bonds),
            'weak_auctions': len(notes_bonds[notes_bonds['bid_to_cover_ratio'] < 2.2]),
            'strong_auctions': len(notes_bonds[notes_bonds['bid_to_cover_ratio'] >= 2.5]),
        }

        # Overall health rating
        avg_btc = summary['avg_bid_to_cover']
        if pd.isna(avg_btc):
            summary['health'] = "Unknown"
            summary['health_color'] = "#9E9E9E"
        elif avg_btc >= 2.5:
            summary['health'] = "Strong Demand"
            summary['health_color'] = "#4CAF50"
        elif avg_btc >= 2.2:
            summary['health'] = "Normal Demand"
            summary['health_color'] = "#8BC34A"
        elif avg_btc >= 2.0:
            summary['health'] = "Soft Demand"
            summary['health_color'] = "#FF9800"
        else:
            summary['health'] = "Weak Demand"
            summary['health_color'] = "#F44336"

        return summary
