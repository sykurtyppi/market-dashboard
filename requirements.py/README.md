# 📊 Market Risk Dashboard

A comprehensive market risk monitoring system that tracks credit spreads, sentiment indicators, and generates trading signals using the LEFT strategy.

## Features

- ✅ **Credit Spread Monitoring** - Track HYG OAS and IG spreads from FRED
- ✅ **LEFT Strategy** - Automated signals based on 330-day EMA
- ✅ **Fear & Greed Index** - Real-time market sentiment
- ✅ **VIX Analysis** - Term structure and contango tracking
- ✅ **Market Breadth** - NYSE/Nasdaq advance-decline data
- ✅ **SQLite Database** - Historical data storage
- ✅ **Streamlit Dashboard** - Beautiful, interactive UI

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/market-dashboard.git
cd market-dashboard

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your FRED API key
```

## Configuration

1. **Get FREE FRED API Key:**
   - Visit https://fred.stlouisfed.org/
   - Sign up (free)
   - Get API key from https://fredaccount.stlouisfed.org/apikeys

2. **Add to .env file:**
```
   FRED_API_KEY=your_key_here
```

## Usage

### Run Initial Data Collection
```bash
python scheduler/daily_update.py
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

Open browser to http://localhost:8501

### Test Individual Components
```bash
# Test FRED collector
python data_collectors/fred_collector.py

# Test LEFT strategy
python processors/left_strategy.py

# Test database
python database/db_manager.py

# Run all tests
python test_collectors.py
```

## Project Structure
```
market-dashboard/
├── data_collectors/     # Data fetching modules
├── processors/          # Signal processing
├── database/           # Data storage
├── dashboard/          # Streamlit UI
├── scheduler/          # Automated updates
├── config/             # Configuration
└── data/               # Database files (auto-created)
```

## LEFT Strategy

The LEFT (Leveraged ETF Trading) strategy uses credit spreads to time market entries:

- **BUY Signal**: Credit spreads fall 35% below 330-day EMA
- **SELL Signal**: Credit spreads rise 40% above 330-day EMA
- **NEUTRAL**: Between thresholds

Based on research from Gilchrist & Zakrajšek (2012) - Credit Spreads and Business Cycle Fluctuations

## Data Sources

- **FRED** - Federal Reserve Economic Data (free)
- **CNN** - Fear & Greed Index (free, scraped)
- **CBOE** - VIX data (free, scraped)
- **WSJ** - Market breadth (free, scraped)

## Automation

Set up daily updates with cron (Linux/Mac):
```bash
# Edit crontab
crontab -e

# Add this line (runs at 4:30 PM EST daily)
30 16 * * 1-5 cd /path/to/market-dashboard && /usr/bin/python3 scheduler/daily_update.py
```

Windows Task Scheduler:
- Create task to run `python scheduler/daily_update.py` at 4:30 PM weekdays

## Disclaimer

**THIS IS NOT FINANCIAL ADVICE**

This tool is for educational and informational purposes only. Always do your own research and consult with financial professionals before making investment decisions.

## License

MIT License - see LICENSE file

## Contributing

Pull requests welcome! Please open an issue first to discuss changes.

## Acknowledgments

- FRED API by Federal Reserve Bank of St. Louis
- CNN Fear & Greed Index
- Research by Gilchrist & Zakrajšek on credit spreads