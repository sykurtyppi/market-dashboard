"""
Insider Trading Collector
Fetches Form 4 filings from SEC EDGAR

Insider transactions can signal smart money sentiment:
- Cluster buying by multiple insiders = Bullish signal
- Large sales (especially by CEO/CFO) = Potential red flag
- Open market purchases > Option exercises (stronger signal)

Data source: SEC EDGAR (free, public data)

Key transaction codes:
- P = Open market purchase (strongest bullish signal)
- S = Open market sale
- A = Grant/award
- M = Exercise of derivative
- G = Gift
- F = Tax withholding (usually automatic, not directional)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import re
from utils.retry_utils import exponential_backoff_retry

logger = logging.getLogger(__name__)

# Tracked company CIKs (SEC identifiers) for major companies
# CIK lookup: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
TRACKED_COMPANIES = {
    'AAPL': '0000320193',
    'MSFT': '0000789019',
    'GOOGL': '0001652044',
    'AMZN': '0001018724',
    'NVDA': '0001045810',
    'META': '0001326801',
    'TSLA': '0001318605',
    'JPM': '0000019617',
    'BAC': '0000070858',
    'XOM': '0000034088',
    'V': '0001403161',
    'JNJ': '0000200406',
    'WMT': '0000104169',
    'PG': '0000080424',
    'MA': '0001141391',
    'HD': '0000354950',
    'DIS': '0001744489',
    'NFLX': '0001065280',
    'AMD': '0000002488',
    'CRM': '0001108524',
}


class InsiderTradingCollector:
    """
    Collects insider trading data from SEC EDGAR

    Tracks Form 4 filings which report insider transactions
    within 2 business days of the trade.

    Uses SEC EDGAR's public RSS feed and submissions API.
    Rate limit: 10 requests/second (SEC requirement)
    """

    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.headers = {
            # SEC requires a valid User-Agent with contact info
            'User-Agent': 'MarketDashboard/1.0 (market-dashboard@example.com)',
            'Accept': 'application/json, application/xml, text/html'
        }
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 900  # 15 minutes cache to respect SEC rate limits

    def _get_ticker_from_cik(self, cik: str) -> Optional[str]:
        """Reverse lookup ticker from CIK"""
        for ticker, company_cik in TRACKED_COMPANIES.items():
            if company_cik.lstrip('0') == cik.lstrip('0'):
                return ticker
        return None

    @exponential_backoff_retry(max_retries=2, base_delay=1.0)
    def get_recent_filings(self, cik: str = None, days: int = 30) -> Optional[List[Dict]]:
        """
        Fetch recent Form 4 insider trading filings from SEC EDGAR

        Args:
            cik: Optional company CIK to filter
            days: Number of days to look back

        Returns:
            List of insider transaction dictionaries
        """
        # Check cache first
        cache_key = f"filings_{cik or 'all'}_{days}"
        if self._cache_time and (datetime.now() - self._cache_time).seconds < self._cache_ttl:
            if cache_key in self._cache:
                return self._cache[cache_key]

        all_filings = []

        # If specific CIK provided, fetch just that one
        if cik:
            filings = self._fetch_company_filings(cik, days)
            if filings:
                all_filings.extend(filings)
        else:
            # Fetch from multiple tracked companies (limit to avoid rate limiting)
            companies_to_fetch = list(TRACKED_COMPANIES.items())[:10]  # Top 10

            for ticker, company_cik in companies_to_fetch:
                try:
                    filings = self._fetch_company_filings(company_cik, days)
                    if filings:
                        for f in filings:
                            f['ticker'] = ticker
                        all_filings.extend(filings)
                except Exception as e:
                    logger.debug(f"Error fetching {ticker}: {e}")
                    continue

        if not all_filings:
            logger.info("No live SEC data, using recent sample data")
            return self._get_sample_insider_data()

        # Sort by date descending
        all_filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)

        # Cache results
        self._cache[cache_key] = all_filings[:30]
        self._cache_time = datetime.now()

        return all_filings[:30]

    def _fetch_company_filings(self, cik: str, days: int) -> List[Dict]:
        """Fetch Form 4 filings for a specific company"""
        try:
            url = f"{self.base_url}/submissions/CIK{cik.zfill(10)}.json"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 403:
                logger.warning(f"SEC rate limited for CIK {cik}")
                return []

            if response.status_code != 200:
                return []

            data = response.json()
            filings = []

            recent = data.get('filings', {}).get('recent', {})
            if not recent:
                return []

            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            primary_docs = recent.get('primaryDocument', [])

            cutoff_date = datetime.now() - timedelta(days=days)
            company_name = data.get('name', 'Unknown')

            for i, form in enumerate(forms):
                if form == '4' and i < len(dates):
                    try:
                        filing_date = datetime.strptime(dates[i], '%Y-%m-%d')
                        if filing_date >= cutoff_date:
                            filings.append({
                                'form': form,
                                'filing_date': dates[i],
                                'accession': accessions[i] if i < len(accessions) else None,
                                'primary_doc': primary_docs[i] if i < len(primary_docs) else None,
                                'company': company_name,
                                'cik': data.get('cik', cik),
                                'ticker': self._get_ticker_from_cik(cik),
                            })
                    except (ValueError, IndexError):
                        continue

            return filings[:10]  # Max 10 per company

        except Exception as e:
            logger.debug(f"Error fetching CIK {cik}: {e}")
            return []

    def _get_sample_insider_data(self) -> List[Dict]:
        """
        Provide sample insider trading data when SEC API unavailable

        This represents typical insider activity patterns based on
        historical Form 4 filing patterns.

        Note: In production, this would be replaced with cached
        recent data from successful API calls.
        """
        today = datetime.now()

        # Sample realistic insider transactions
        # Reflects typical patterns: CEO sales on 10b5-1 plans,
        # director purchases (more bullish signal), option exercises
        sample_data = [
            {
                'company': 'Apple Inc.',
                'ticker': 'AAPL',
                'insider_name': 'Tim Cook',
                'title': 'CEO',
                'transaction_type': 'Sale (10b5-1 Plan)',
                'shares': 50000,
                'price': 185.50,
                'value': 9275000,
                'date': (today - timedelta(days=3)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=2)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'S',
                'is_10b5_1': True,
                'data_source': 'sample',
            },
            {
                'company': 'Microsoft Corp.',
                'ticker': 'MSFT',
                'insider_name': 'Satya Nadella',
                'title': 'CEO',
                'transaction_type': 'Sale (10b5-1 Plan)',
                'shares': 10000,
                'price': 415.00,
                'value': 4150000,
                'date': (today - timedelta(days=5)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=4)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'S',
                'is_10b5_1': True,
                'data_source': 'sample',
            },
            {
                'company': 'NVIDIA Corp.',
                'ticker': 'NVDA',
                'insider_name': 'Mark Stevens',
                'title': 'Director',
                'transaction_type': 'Open Market Purchase',
                'shares': 1000,
                'price': 875.00,
                'value': 875000,
                'date': (today - timedelta(days=7)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=6)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'P',
                'is_10b5_1': False,
                'data_source': 'sample',
            },
            {
                'company': 'Tesla Inc.',
                'ticker': 'TSLA',
                'insider_name': 'Robyn Denholm',
                'title': 'Director/Chair',
                'transaction_type': 'Open Market Purchase',
                'shares': 5000,
                'price': 245.00,
                'value': 1225000,
                'date': (today - timedelta(days=10)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=9)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'P',
                'is_10b5_1': False,
                'data_source': 'sample',
            },
            {
                'company': 'JPMorgan Chase',
                'ticker': 'JPM',
                'insider_name': 'Jamie Dimon',
                'title': 'CEO',
                'transaction_type': 'Open Market Purchase',
                'shares': 25000,
                'price': 195.00,
                'value': 4875000,
                'date': (today - timedelta(days=12)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=11)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'P',
                'is_10b5_1': False,
                'data_source': 'sample',
            },
            {
                'company': 'Amazon.com Inc.',
                'ticker': 'AMZN',
                'insider_name': 'Andrew Jassy',
                'title': 'CEO',
                'transaction_type': 'Sale (10b5-1 Plan)',
                'shares': 8000,
                'price': 178.00,
                'value': 1424000,
                'date': (today - timedelta(days=14)).strftime('%Y-%m-%d'),
                'filing_date': (today - timedelta(days=13)).strftime('%Y-%m-%d'),
                'form_type': '4',
                'transaction_code': 'S',
                'is_10b5_1': True,
                'data_source': 'sample',
            },
        ]

        return sample_data

    def get_insider_summary(self, days: int = 30) -> Dict:
        """
        Get summary of recent insider activity

        Returns:
            Dict with aggregate insider trading metrics

        Key insight for interpretation:
        - Open market purchases (P) are strongest bullish signal
        - 10b5-1 plan sales are often pre-scheduled, less meaningful
        - Cluster buying (multiple insiders) is very bullish
        - CEO/CFO purchases carry more weight
        """
        transactions = self.get_recent_filings(days=days)

        if not transactions:
            return {
                'status': 'unavailable',
                'message': 'Insider trading data not available'
            }

        # Determine data source
        is_sample = any(t.get('data_source') == 'sample' for t in transactions)

        # Count buys vs sells (only count P and S transactions, not exercises)
        buys = [t for t in transactions if t.get('transaction_code') == 'P']
        sells = [t for t in transactions if t.get('transaction_code') == 'S']

        # Separate 10b5-1 planned sales (less meaningful)
        planned_sells = [t for t in sells if t.get('is_10b5_1', False)]
        discretionary_sells = [t for t in sells if not t.get('is_10b5_1', False)]

        total_buy_value = sum(t.get('value', 0) for t in buys)
        total_sell_value = sum(t.get('value', 0) for t in sells)
        discretionary_sell_value = sum(t.get('value', 0) for t in discretionary_sells)

        # Calculate buy/sell ratio (use discretionary for more accurate signal)
        if discretionary_sell_value > 0:
            buy_sell_ratio = total_buy_value / discretionary_sell_value
        elif total_sell_value > 0:
            buy_sell_ratio = total_buy_value / total_sell_value
        else:
            buy_sell_ratio = float('inf') if total_buy_value > 0 else 1.0

        # Get unique tickers
        tickers_with_activity = set(t.get('ticker') for t in transactions if t.get('ticker'))

        summary = {
            'status': 'sample' if is_sample else 'live',
            'data_source': 'SEC EDGAR Form 4' if not is_sample else 'Sample Data',
            'period_days': days,
            'total_transactions': len(transactions),
            'buy_count': len(buys),
            'sell_count': len(sells),
            'planned_sell_count': len(planned_sells),
            'discretionary_sell_count': len(discretionary_sells),
            'total_buy_value': total_buy_value,
            'total_sell_value': total_sell_value,
            'buy_sell_ratio': round(buy_sell_ratio, 2) if buy_sell_ratio != float('inf') else 999,
            'sentiment': self._interpret_insider_sentiment(buy_sell_ratio),
            'tickers_with_activity': list(tickers_with_activity),
            'last_updated': datetime.now().isoformat(),
        }

        # Add color coding based on buy/sell ratio
        # Note: Ratio > 1 means more buying than selling (bullish)
        if buy_sell_ratio > 2.0 or (len(buys) > 0 and len(discretionary_sells) == 0):
            summary['color'] = '#4CAF50'  # Green - bullish
            summary['signal'] = 'BULLISH'
            summary['signal_strength'] = 'STRONG' if buy_sell_ratio > 3 else 'MODERATE'
        elif buy_sell_ratio > 0.8:
            summary['color'] = '#FFC107'  # Yellow - neutral
            summary['signal'] = 'NEUTRAL'
            summary['signal_strength'] = 'WEAK'
        elif buy_sell_ratio > 0.3:
            summary['color'] = '#FF9800'  # Orange - mildly bearish
            summary['signal'] = 'CAUTIOUS'
            summary['signal_strength'] = 'MODERATE'
        else:
            summary['color'] = '#F44336'  # Red - bearish
            summary['signal'] = 'BEARISH'
            summary['signal_strength'] = 'STRONG' if buy_sell_ratio < 0.1 else 'MODERATE'

        return summary

    def _interpret_insider_sentiment(self, ratio: float) -> str:
        """Interpret insider buy/sell ratio"""
        if ratio == float('inf'):
            return "All buying, no selling - Very bullish signal"
        elif ratio > 2.0:
            return "Strong buying activity - Insiders bullish on their companies"
        elif ratio > 1.0:
            return "More buying than selling - Modestly bullish"
        elif ratio > 0.5:
            return "Mixed activity - No clear directional signal"
        elif ratio > 0.2:
            return "More selling than buying - Some profit taking"
        else:
            return "Heavy selling - Insiders reducing exposure"

    def get_notable_transactions(self, min_value: float = 1000000, days: int = 30) -> List[Dict]:
        """
        Get notable large insider transactions

        Args:
            min_value: Minimum transaction value to include
            days: Number of days to look back

        Returns:
            List of large insider transactions
        """
        transactions = self.get_recent_filings(days=days)

        if not transactions:
            return []

        notable = [t for t in transactions if t.get('value', 0) >= min_value]

        # Sort by value descending
        notable.sort(key=lambda x: x.get('value', 0), reverse=True)

        return notable

    def get_cluster_buying(self, days: int = 30) -> List[Dict]:
        """
        Identify cluster buying patterns

        Cluster buying = Multiple insiders buying same stock
        This is a stronger bullish signal than single insider purchases

        Returns:
            List of stocks with cluster buying activity
        """
        transactions = self.get_recent_filings(days=days)

        if not transactions:
            return []

        # Group by company
        company_buys = {}
        for t in transactions:
            if t.get('transaction_code') == 'P':
                ticker = t.get('ticker', 'Unknown')
                if ticker not in company_buys:
                    company_buys[ticker] = []
                company_buys[ticker].append(t)

        # Find clusters (2+ insiders buying)
        clusters = []
        for ticker, buys in company_buys.items():
            if len(buys) >= 2:
                clusters.append({
                    'ticker': ticker,
                    'company': buys[0].get('company', 'Unknown'),
                    'insider_count': len(buys),
                    'total_value': sum(t.get('value', 0) for t in buys),
                    'signal': 'STRONG BUY SIGNAL',
                    'description': f"{len(buys)} insiders buying within {days} days"
                })

        return clusters

    def get_ceo_cfo_activity(self, days: int = 60) -> List[Dict]:
        """
        Track CEO and CFO transactions specifically

        C-suite transactions carry more weight as these executives
        have the deepest knowledge of company prospects

        Returns:
            List of CEO/CFO transactions
        """
        transactions = self.get_recent_filings(days=days)

        if not transactions:
            return []

        c_suite = [
            t for t in transactions
            if t.get('title', '').upper() in ['CEO', 'CFO', 'PRESIDENT', 'COO']
        ]

        return c_suite
