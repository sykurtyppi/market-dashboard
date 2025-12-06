"""
Technical Indicators Module
Provides EMA, SMA, RSI, Z-Score with robust validation
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
#  SIMPLE MOVING AVERAGE
# ---------------------------------------------------------
def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        series: Price/value series
        window: Number of periods for moving average
    
    Returns:
        Series with SMA values
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    if window <= 0:
        raise ValueError("Window must be positive")
    
    return series.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------
#  EXPONENTIAL MOVING AVERAGE
# ---------------------------------------------------------
def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Price/value series
        window: Span for EMA calculation
    
    Returns:
        Series with EMA values
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    if window <= 0:
        raise ValueError("Window must be positive")
    
    return series.ewm(span=window, adjust=False).mean()


# ---------------------------------------------------------
#  RELATIVE STRENGTH INDEX
# ---------------------------------------------------------
def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        series: Price series
        window: Period for RSI calculation (default 14)
    
    Returns:
        Series with RSI values (0-100)
    """
    if series is None or len(series) < window + 1:
        return pd.Series(dtype=float)
    
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = pd.Series(gain, index=series.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ---------------------------------------------------------
#  Z-SCORE NORMALIZATION
# ---------------------------------------------------------
def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling Z-Score.
    
    Args:
        series: Value series
        window: Lookback period for mean/std calculation
    
    Returns:
        Series with Z-Score values
    """
    if series is None or len(series) < window:
        return pd.Series(dtype=float)
    
    # Calculate rolling mean and std
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=0)
    
    # Calculate Z-Score
    z_score = (series - mean) / std.replace(0, np.nan)
    
    return z_score


# ---------------------------------------------------------
#  BOLLINGER BANDS
# ---------------------------------------------------------
def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series
        window: Period for moving average
        num_std: Number of standard deviations for bands
    
    Returns:
        DataFrame with 'middle', 'upper', 'lower' columns
    """
    if series is None or len(series) < window:
        return pd.DataFrame()
    
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': upper,
        'lower': lower
    }, index=series.index)


# ---------------------------------------------------------
#  MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------
def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD indicator.
    
    Args:
        series: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        DataFrame with 'macd', 'signal', 'histogram' columns
    """
    if series is None or len(series) < slow:
        return pd.DataFrame()
    
    # Calculate EMAs
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=series.index)


# ---------------------------------------------------------
#  PERCENTAGE CHANGE
# ---------------------------------------------------------
def calculate_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate percentage change.
    
    Args:
        series: Value series
        periods: Number of periods to calculate change over
    
    Returns:
        Series with percentage change values
    """
    if series is None or len(series) <= periods:
        return pd.Series(dtype=float)
    
    return series.pct_change(periods=periods) * 100


# ---------------------------------------------------------
#  CUMULATIVE RETURNS
# ---------------------------------------------------------
def calculate_cumulative_returns(series: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from a returns series.
    
    Args:
        series: Returns series (as decimals, not percentages)
    
    Returns:
        Series with cumulative returns
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    
    return (1 + series).cumprod() - 1


# ---------------------------------------------------------
#  DRAWDOWN
# ---------------------------------------------------------
def calculate_drawdown(series: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown from peak.
    
    Args:
        series: Price/value series
    
    Returns:
        DataFrame with 'peak', 'drawdown', 'drawdown_pct' columns
    """
    if series is None or len(series) == 0:
        return pd.DataFrame()
    
    # Calculate running maximum
    running_max = series.expanding().max()
    
    # Calculate drawdown
    drawdown = series - running_max
    drawdown_pct = (drawdown / running_max) * 100
    
    return pd.DataFrame({
        'peak': running_max,
        'drawdown': drawdown,
        'drawdown_pct': drawdown_pct
    }, index=series.index)


# ---------------------------------------------------------
#  VOLATILITY (Annualized)
# ---------------------------------------------------------
def calculate_volatility(series: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation).
    
    Args:
        series: Price series
        window: Rolling window period
        annualize: If True, annualize using √252
    
    Returns:
        Series with volatility values
    """
    if series is None or len(series) < window:
        return pd.Series(dtype=float)
    
    # Calculate log returns
    log_returns = np.log(series / series.shift(1))
    
    # Calculate rolling standard deviation
    volatility = log_returns.rolling(window=window).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility * 100  # Convert to percentage


if __name__ == "__main__":
    # Test the indicators
    import yfinance as yf
    
    print("Testing Technical Indicators Module")
    print("=" * 60)
    
    # Fetch test data
    spy = yf.Ticker("SPY")
    data = spy.history(period="1y")
    prices = data['Close']
    
    print(f"\nTest data: {len(prices)} days of SPY prices")
    print(f"Latest price: ${prices.iloc[-1]:.2f}")
    
    # Test SMA
    sma_20 = calculate_sma(prices, 20)
    print(f"\nSMA(20): ${sma_20.iloc[-1]:.2f}")
    
    # Test EMA
    ema_20 = calculate_ema(prices, 20)
    print(f"EMA(20): ${ema_20.iloc[-1]:.2f}")
    
    # Test RSI
    rsi = calculate_rsi(prices, 14)
    print(f"RSI(14): {rsi.iloc[-1]:.2f}")
    
    # Test Z-Score
    z_score = calculate_z_score(prices, 20)
    print(f"Z-Score(20): {z_score.iloc[-1]:.2f}")
    
    # Test Volatility
    vol = calculate_volatility(prices, 21)
    print(f"Volatility(21d): {vol.iloc[-1]:.2f}%")
    
    print("\n✅ All indicators working!")