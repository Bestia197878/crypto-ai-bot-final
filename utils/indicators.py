"""
Technical Indicators Library
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from loguru import logger


class TechnicalIndicators:
    """
    Comprehensive technical indicators library
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int = 20) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > abs(low.diff())), 0)
        
        # Minus Directional Movement (-DM)
        minus_dm = -low.diff()
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm.shift()), 0)
        
        # Smooth +DM and -DM
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Directional Movement Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.replace([np.inf, -np.inf], 0)
        
        # Average Directional Index (ADX)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def fibonacci_retracement(
        high: float,
        low: float,
        levels: List[float] = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    ) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        retracements = {}
        
        for level in levels:
            retracements[level] = high - (diff * level)
        
        return retracements
    
    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Calculate pivot points"""
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }
    
    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        prefix: str = ""
    ) -> pd.DataFrame:
        """Calculate all indicators for a DataFrame"""
        result = df.copy()
        
        # Moving averages
        result[f'{prefix}sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        result[f'{prefix}sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        result[f'{prefix}ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        result[f'{prefix}ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # Momentum
        result[f'{prefix}rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        
        # MACD
        macd, signal, hist = TechnicalIndicators.macd(df['close'])
        result[f'{prefix}macd'] = macd
        result[f'{prefix}macd_signal'] = signal
        result[f'{prefix}macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
        result[f'{prefix}bb_upper'] = upper
        result[f'{prefix}bb_middle'] = middle
        result[f'{prefix}bb_lower'] = lower
        
        # Volatility
        result[f'{prefix}atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        result[f'{prefix}stoch_k'] = k
        result[f'{prefix}stoch_d'] = d
        
        # ADX
        adx, plus_di, minus_di = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        result[f'{prefix}adx'] = adx
        result[f'{prefix}plus_di'] = plus_di
        result[f'{prefix}minus_di'] = minus_di
        
        # Volume
        result[f'{prefix}obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        result[f'{prefix}vwap'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        return result
