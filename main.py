"""
Ultra-Robust Narrative Analyzer - No external API dependencies
"""

import time
import math
import json
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional,Tuple
import numpy as np
import pandas as pd
import ccxt
from scipy.stats import linregress
import matplotlib.pyplot as plt
import requests

# --------------------------------------------------
# ULTRA-ROBUST CONFIGURATION
# --------------------------------------------------
CONFIG = {
    "exchange": "binance",
    "fiat": "USDT",
    "symbols": ["ASTER/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT", "LTC/USDT", "TRX/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT", "TRUMP/USDT"],
    "timeframe": "1d",
    "lookback_days": 90,
    "recent_window": 3,
    "prev_window": 14,
    "min_ohlcv_points": 20,
    "save_folder": "narrative_output",
    
    # Optimized weights for OHLCV-only analysis
    "weights": {
        "price_action": 2.5,
        "volume_analysis": 2.0,
        "market_structure": 2.0,
        "momentum": 1.8,
        "volatility": 1.5,
        "market_sentiment": 1.2,
    },
    
    "logistic_k": 2.0,
    "logistic_x0": 0.0,
    
    # --- Мультивременные настройки ---
    "htf_timeframe": "1w",           # Higher Timeframe
    "ltf_timeframe": "4h",           # Lower Timeframe
    "htf_lookback": 180,             # Дней для HTF
    "ltf_lookback": 30,              # Дней для LTF

    # --- Режимы рынка ---
    "market_regime": {
        "trend_threshold": 0.6,      # ADX > 25 → тренд
        "consolidation_threshold": 0.3,
    },

    # --- Волатильность ---
    "volatility_regime": {
        "high_vol_threshold": 0.08,  # >8% → высокая волатильность
        "low_vol_threshold": 0.02,   # <2% → низкая
        "atr_filter_enabled": True,
    },

    # --- Ликвидность ---
    "liquidity_filter": {
        "min_volume_usd": 10_000_000,  # Только ликвидные
        "volume_ma_days": 20,
    },

    # --- Новые веса ---
    "weights_v2": {
        "htf_alignment": 3.0,
        "volatility_regime": 2.2,
        "market_regime": 2.0,
        "liquidity_score": 1.5,
    },
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.makedirs(CONFIG["save_folder"], exist_ok=True)

COINGECKO_CACHE_DIR = os.path.join(CONFIG["save_folder"], "cache")
os.makedirs(COINGECKO_CACHE_DIR, exist_ok=True)

def load_from_cache(coin_id: str, days: int) -> Optional[dict]:
    """Загружает кэшированные данные, если они не старше 24 часов."""
    cache_file = os.path.join(COINGECKO_CACHE_DIR, f"{coin_id}_{days}.json")
    if os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < 24 * 3600:  # < 24 часов
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
    return None

def save_to_cache(coin_id: str, days: int, data: dict):
    """Сохраняет данные в кэш."""
    cache_file = os.path.join(COINGECKO_CACHE_DIR, f"{coin_id}_{days}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    
# --------------------------------------------------
# ENHANCED TECHNICAL ANALYSIS ENGINE
# --------------------------------------------------
class VolatilityRegimeEngine:
    """Анализ волатильности как фильтра и драйвера"""
    
    def score_volatility_regime(self, indicators: Dict, mtf: Dict) -> float:
        score = 0
        factors = 0
        
        regime = mtf.get('volatility_regime', 'normal')
        
        # Высокая волатильность + тренд = хорошо
        if regime == 'high' and mtf.get('market_regime') == 'trending':
            score += 0.8
        elif regime == 'high' and indicators.get('bb_position', 0.5) > 0.9:
            score -= 0.6  # Пробой вверх в хай-воле — риск
        factors += 1
        
        # Низкая волатильность + сжатие = накопление
        atr_pct = indicators.get('atr_percent', 0.02)
        if regime == 'low' and atr_pct < 0.015:
            score += 0.7  # Потенциальный breakout
        factors += 1
        
        # Волатильность vs средняя
        current_vol = indicators.get('volatility_20d', 0.02)
        hist_vol = pd.Series([indicators.get(f'volatility_{i}d', 0.02) for i in [20, 50, 100]]).mean()
        if current_vol > hist_vol * 1.5:
            score -= 0.4  # Аномальный всплеск
        factors += 1
        
        return score / factors
    
    
class MultiTimeframeAnalyzer:
    """Анализ согласованности между HTF и LTF"""
    
    def __init__(self, data_manager):
        self.dm = data_manager
    
    def analyze_mtf(self, symbol: str) -> Dict[str, Any]:
        tf_configs = [
            ("1d", CONFIG["lookback_days"]),
            (CONFIG["htf_timeframe"], CONFIG["htf_lookback"]),
            (CONFIG["ltf_timeframe"], CONFIG["ltf_lookback"]),
        ]
        
        data = {}
        for tf, days in tf_configs:
            df = self.dm.fetch_ohlcv(symbol, tf, days)
            if not df.empty and len(df) >= 10:
                data[tf] = df
            else:
                data[tf] = None
        
        if all(df is None or df.empty for df in data.values()):
            return {}
        
        # Получаем daily данные для расчета regime
        daily_df = data.get("1d")
        if daily_df is None:
            regime_score = 0.5
            regime_label = 'mixed'
        else:
            regime_score = self._get_market_regime(daily_df)
            regime_label = 'trending' if regime_score > 0.6 else 'consolidation' if regime_score < 0.4 else 'mixed'
        
        result = {
            'htf_trend': self._get_trend_direction(data.get(CONFIG["htf_timeframe"])),
            'ltf_trend': self._get_trend_direction(data.get(CONFIG["ltf_timeframe"])),
            'daily_trend': self._get_trend_direction(data.get("1d")),
            'htf_rsi': self._get_rsi(data.get(CONFIG["htf_timeframe"])),
            'ltf_rsi': self._get_rsi(data.get(CONFIG["ltf_timeframe"])),
            'volatility_regime': self._get_volatility_regime(data.get("1d")),
            'liquidity_score': self._get_liquidity_score(data.get("1d"), symbol),
            'market_regime_score': regime_score,  # числовая оценка
            'market_regime': regime_label,        # строковая метка
        }
        
        result['htf_alignment_score'] = self._calculate_alignment(result)
        return result
    
    def _get_trend_direction(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 20: return 0
        close = df['close']
        ema20 = ema(close, 20).iloc[-1]
        ema50 = ema(close, 50).iloc[-1]
        slope = (close.iloc[-1] - close.iloc[-10]) / 10
        return 1.0 if (ema20 > ema50 and slope > 0) else -1.0 if (ema20 < ema50 and slope < 0) else 0.0
    
    def _get_rsi(self, df: pd.DataFrame) -> float:
        if df is None or df.empty: return 50.0
        return rsi(df['close'], 14).iloc[-1]
    
    def _get_volatility_regime(self, df: pd.DataFrame) -> str:
        if df is None: return "unknown"
        vol = df['close'].pct_change().rolling(20).std().iloc[-1]
        if vol > CONFIG["volatility_regime"]["high_vol_threshold"]:
            return "high"
        elif vol < CONFIG["volatility_regime"]["low_vol_threshold"]:
            return "low"
        else:
            return "normal"
    def _get_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple methods"""
        if df is None or len(df) < 20:
            return 0.0
        
        try:
            close = df['close']
            
            # Method 1: Linear regression slope
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            
            # Проверяем, что есть достаточное количество данных
            if len(y) < 5:
                return 0.0
                
            slope, _, r_value, _, _ = linregress(x, y)
            regression_strength = abs(slope) * (r_value ** 2)
            
            # Method 2: EMA alignment
            ema_20 = ema(close, 20).iloc[-1]
            ema_50 = ema(close, 50).iloc[-1]
            
            # Защита от деления на ноль
            current_price = close.iloc[-1]
            if current_price == 0:
                ema_ratio = 0
            else:
                ema_ratio = abs(ema_20 - ema_50) / current_price
            
            # Method 3: Price momentum
            if len(close) >= 6:
                momentum_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
            else:
                momentum_5 = 0
                
            if len(close) >= 11:
                momentum_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            else:
                momentum_10 = 0
                
            momentum_strength = (abs(momentum_5) + abs(momentum_10)) / 2
            
            # Combine methods (с защитой от деления на ноль)
            combined_strength = 0
            valid_methods = 0
            
            if not np.isnan(regression_strength):
                combined_strength += regression_strength * 1000
                valid_methods += 1
                
            if not np.isnan(ema_ratio):
                combined_strength += ema_ratio * 100
                valid_methods += 1
                
            if not np.isnan(momentum_strength):
                combined_strength += momentum_strength
                valid_methods += 1
            
            if valid_methods > 0:
                combined_strength = combined_strength / valid_methods
            else:
                combined_strength = 0
            
            # Determine direction
            direction = 1.0 if (ema_20 > ema_50 and slope > 0) else -1.0 if (ema_20 < ema_50 and slope < 0) else 0.0
            
            return combined_strength * direction
            
        except Exception as e:
            logger.warning(f"Error calculating trend strength: {e}")
            return 0.0
    def _get_market_regime(self, df: pd.DataFrame) -> float:
        """Calculate market regime score (0-1)"""
        if df is None or len(df) < 25:
            return 0.5
        
        try:
            adx = self._calculate_adx(df)
            trend_strength = self._get_trend_strength(df)
            
            # Normalize ADX (0-1 range)
            adx_normalized = min(adx / 50.0, 1.0)  # ADX > 50 is very strong
            
            # Use trend strength magnitude (0-1 range)
            trend_magnitude = min(abs(trend_strength) * 10, 1.0)  # Scale appropriately
            
            # Combine ADX and trend strength
            if adx > 25 and trend_magnitude > 0.3:
                # Strong trending regime
                regime_strength = 0.3 + (adx_normalized * 0.7)
                return regime_strength if trend_strength > 0 else 1.0 - regime_strength
            elif adx < 15 and trend_magnitude < 0.2:
                # Consolidation regime
                return 0.5
            else:
                # Mixed/weak trend
                base = 0.5
                adjustment = trend_magnitude * 0.4 * (1 if trend_strength > 0 else -1)
                return max(0.1, min(0.9, base + adjustment))
                
        except Exception as e:
            logger.warning(f"Error calculating market regime: {e}")
            return 0.5
    
    def _calculate_adx(self, df: pd.DataFrame, period=14) -> float:
        high, low, close = df['high'], df['low'], df['close']
        plus_di = 100 * (high.diff(1).ewm(alpha=1/period).mean() / 
                         calculate_atr_df(df, period))
        minus_di = 100 * (low.diff(1).abs().ewm(alpha=1/period).mean() / 
                          calculate_atr_df(df, period))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=1/period).mean().iloc[-1]
        return adx if not pd.isna(adx) else 0
    
    def _get_liquidity_score(self, df: pd.DataFrame, symbol: str) -> float:
        if df is None: return 0
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        price = df['close'].iloc[-1]
        vol_usd = avg_vol * price
        
        # Логарифмическая шкала для более реалистичной оценки
        min_vol = CONFIG["liquidity_filter"]["min_volume_usd"]
        max_vol = 1_000_000_000  # $1B как максимум
        vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
        vol_ma5 = df['volume'].tail(5).mean()
        vol_trend = (vol_ma5 / vol_ma20) if vol_ma20 > 0 else 1.0
        
        usd_vol = vol_ma20 * df['close'].iloc[-1]
        # Нормализация в диапазоне 0-1 с логарифмом
        if vol_usd <= min_vol:
            return 0.1
        else:
            log_score = (math.log10(vol_usd) - math.log10(min_vol)) / \
                    (math.log10(max_vol) - math.log10(min_vol))
            return 0.7 * log_score + 0.3 * min(vol_trend, 2.0)  # cap at 2x
    
    def _calculate_alignment(self, mtf: Dict) -> float:
        score = 0
        factors = 0
        
        # HTF и Daily согласованность
        htf_trend = mtf.get('htf_trend', 0)
        daily_trend = mtf.get('daily_trend', 0)
        
        # Используем допуск для сравнения float
        if abs(htf_trend - daily_trend) < 0.1 and htf_trend != 0:
            score += 1.0
        elif htf_trend * daily_trend < -0.1:  # Противоположные направления
            score -= 0.7
        factors += 1
        
        # LTF подтверждает?
        ltf_trend = mtf.get('ltf_trend', 0)
        if abs(ltf_trend - daily_trend) < 0.1 and ltf_trend != 0:
            score += 0.6
        factors += 1
        
        # RSI согласованность
        htf_rsi = mtf.get('htf_rsi', 50)
        ltf_rsi = mtf.get('ltf_rsi', 50)
        if abs(htf_rsi - ltf_rsi) < 15:
            score += 0.4
        factors += 1
        
        return score / max(factors, 1) if factors > 0 else 0
    
class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis using only OHLCV data"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        
        # 1. PRICE ACTION INDICATORS
        indicators.update(self._calculate_price_action(close, high, low))
        
        # 2. VOLUME ANALYSIS
        indicators.update(self._calculate_volume_analysis(close, volume))
        
        # 3. MOMENTUM INDICATORS
        indicators.update(self._calculate_momentum(close, high, low))
        
        # 4. VOLATILITY ANALYSIS
        indicators.update(self._calculate_volatility(close, high, low))
        
        # 5. MARKET STRUCTURE
        indicators.update(self._calculate_market_structure(close, high, low, symbol))
        
        # 6. SUPPORT/RESISTANCE
        indicators.update(self._calculate_support_resistance(close, high, low))
        
        return indicators
    
    def _calculate_price_action(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Price action analysis"""
        indicators = {}
        
        # Current price position
        current_price = close.iloc[-1]
        sma_20 = sma(close, 20).iloc[-1]
        sma_50 = sma(close, 50).iloc[-1]
        
        indicators['price_vs_sma20'] = (current_price - sma_20) / sma_20
        indicators['price_vs_sma50'] = (current_price - sma_50) / sma_50
        
        # Trend alignment
        indicators['trend_alignment'] = 1.0 if sma_20 > sma_50 else -1.0
        
        # Price channels
        high_20 = high.rolling(20).max().iloc[-1]
        low_20 = low.rolling(20).min().iloc[-1]
        if high_20 != low_20:
            indicators['channel_position'] = (current_price - low_20) / (high_20 - low_20)
        else:
            indicators['channel_position'] = 0.5
        
        return indicators
    
    def _calculate_volume_analysis(self, close: pd.Series, volume: pd.Series) -> Dict[str, float]:
        """Volume analysis"""
        indicators = {}
        
        # Volume trends
        volume_ma_20 = sma(volume, 20).iloc[-1]
        current_volume = volume.iloc[-1]
        indicators['volume_ratio'] = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
        
        # Volume momentum
        volume_5 = volume.tail(5).mean()
        volume_20 = volume.tail(20).mean()
        indicators['volume_momentum'] = (volume_5 - volume_20) / volume_20 if volume_20 > 0 else 0
        
        # Volume-price correlation
        if len(volume) >= 10:
            price_changes = close.pct_change().tail(10)
            volume_changes = volume.pct_change().tail(10)
            correlation = price_changes.corr(volume_changes)
            indicators['volume_price_corr'] = 0 if pd.isna(correlation) else correlation
        
        # OBV (On-Balance Volume)
        obv = calculate_obv(close, volume)
        if len(obv) >= 5:
            indicators['obv_trend'] = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if obv.iloc[-5] != 0 else 0
        
        return indicators
    
    def _calculate_momentum(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Momentum indicators"""
        indicators = {}
        
        # RSI
        rsi_14 = rsi(close, 14).iloc[-1]
        indicators['rsi'] = rsi_14
        
        # Price momentum
        indicators['momentum_5d'] = close.pct_change(5).iloc[-1]
        indicators['momentum_10d'] = close.pct_change(10).iloc[-1]
        indicators['momentum_20d'] = close.pct_change(20).iloc[-1]
        
        # Rate of Change
        roc_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) >= 10 else 0
        indicators['roc_10'] = roc_10
        
        # Stochastic
        stoch_k, stoch_d = stochastic(high, low, close, 14)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        return indicators
    
    def _calculate_volatility(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Volatility analysis"""
        indicators = {}
        
        # Historical volatility
        returns = close.pct_change()
        indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1]
        
        # ATR (Average True Range)
        atr_14 = calculate_atr_df(pd.DataFrame({'high': high, 'low': low, 'close': close}), 14).iloc[-1]
        indicators['atr_percent'] = atr_14 / close.iloc[-1]
        
        # Bollinger Band position
        bb_upper, bb_middle, bb_lower = bollinger_bands(close, 20, 2)
        if bb_upper.iloc[-1] != bb_lower.iloc[-1]:
            indicators['bb_position'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        else:
            indicators['bb_position'] = 0.5
        
        return indicators
    
    def _calculate_market_structure(self, close: pd.Series, high: pd.Series, low: pd.Series, symbol: str) -> Dict[str, float]:
        """Market structure analysis"""
        indicators = {}
        
        # Higher Highs/Lower Lows analysis
        if len(close) >= 10:
            recent_high = high.tail(5).max()
            recent_low = low.tail(5).min()
            prev_high = high.tail(10).head(5).max()
            prev_low = low.tail(10).head(5).min()
            
            # Uptrend: higher highs and higher lows
            if recent_high > prev_high and recent_low > prev_low:
                indicators['market_structure'] = 1.0
            # Downtrend: lower highs and lower lows
            elif recent_high < prev_high and recent_low < prev_low:
                indicators['market_structure'] = -1.0
            else:
                indicators['market_structure'] = 0.0
        
        # Trend strength using linear regression
        if len(close) >= 20:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            slope, _, r_value, _, _ = linregress(x, y)
            indicators['trend_strength'] = slope * 1000  # Scale for better scoring
            indicators['trend_consistency'] = r_value ** 2
        
        return indicators
    
    def _calculate_support_resistance(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict[str, float]:
        """Support and resistance analysis"""
        indicators = {}
        
        if len(close) < 20:
            return indicators
        
        current_price = close.iloc[-1]
        
        # Recent support and resistance levels
        support_level = low.tail(20).min()
        resistance_level = high.tail(20).max()
        
        # Distance to key levels
        price_range = resistance_level - support_level
        if price_range > 0:
            indicators['sr_distance'] = (current_price - support_level) / price_range
        else:
            indicators['sr_distance'] = 0.5
        
        # Breakout potential
        if current_price > resistance_level * 0.98:
            indicators['breakout_potential'] = 1.0
        elif current_price < support_level * 1.02:
            indicators['breakout_potential'] = -1.0
        else:
            indicators['breakout_potential'] = 0.0
        
        return indicators

class IntelligentScoringEngine:
    """Intelligent scoring system using technical indicators as proxy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tech_analyzer = AdvancedTechnicalAnalyzer()
        self.vol_regime = VolatilityRegimeEngine()
    
    def analyze_symbol(self, df: pd.DataFrame, symbol: str, data_manager) -> Dict[str, Any]:
        """Analyze single symbol"""
        if df.empty:
            return None
        
        # Get multi-timeframe analysis
        mtf = MultiTimeframeAnalyzer(data_manager).analyze_mtf(symbol)
        
        # Calculate technical indicators
        indicators = self.tech_analyzer.calculate_all_indicators(df, symbol)
        
        # Calculate category scores
        # В методе analyze_symbol замените:
        scores = {
            'price_action': self._score_price_action(indicators),
            'volume_analysis': self._score_volume_analysis(indicators),
            'market_structure': self._score_market_structure(indicators),
            'momentum': self._score_momentum(indicators),
            'volatility': self._score_volatility(indicators),
            'market_sentiment': self._score_market_sentiment(indicators, symbol),
            'htf_alignment': mtf.get('htf_alignment_score', 0),
            'volatility_regime': self.vol_regime.score_volatility_regime(indicators, mtf),
            'market_regime': mtf.get('market_regime_score', 0.5),  # Используем числовую оценку
            'liquidity_score': mtf.get('liquidity_score', 0),
        }
        
        # Calculate final score
        final_score = self._calculate_final_score_v2(scores)
        
        # Calculate probability
        prob_long = self._calculate_probability(final_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(indicators, scores)
        
        # Get current market data
        current_price = df['close'].iloc[-1]
        price_change_24h = df['close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        volume_24h = df['volume'].iloc[-1]
        
        return {
            'symbol': symbol,
            'final_score': final_score,
            'prob_long': prob_long,
            'confidence': confidence,
            'price': current_price,
            'price_change_24h': price_change_24h,
            'volume_24h': volume_24h,
            'category_scores': scores,
            'indicators': indicators,
        }
    
    def _score_price_action(self, indicators: Dict[str, float]) -> float:
        """Score price action"""
        score = 0
        factors = 0
        
        # Price vs SMAs
        price_vs_sma20 = indicators.get('price_vs_sma20', 0)
        if price_vs_sma20 > 0.02:
            score += 0.4
        elif price_vs_sma20 < -0.02:
            score -= 0.4
        factors += 1
        
        # Channel position
        channel_pos = indicators.get('channel_position', 0.5)
        if channel_pos < 0.3:
            score += 0.3  # Near support
        elif channel_pos > 0.7:
            score -= 0.3  # Near resistance
        factors += 1
        
        sr_dist = indicators.get('sr_distance', 0.5)
        if sr_dist < 0.2:
            score += 0.4  # Близко к поддержке
        elif sr_dist > 0.8:
            score -= 0.4  # Близко к сопротивлению
        factors += 1
        return score / factors if factors > 0 else 0
    
    def _score_volume_analysis(self, indicators: Dict[str, float]) -> float:
        """Score volume indicators"""
        score = 0
        factors = 0
        
        # Volume ratio
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            score += 0.4
        elif volume_ratio < 0.8:
            score -= 0.2
        factors += 1
        
        
        obv = indicators.get('obv_trend', 0)
        if obv > 0.1: score += 0.3
        elif obv < -0.1: score -= 0.3
        factors += 1
        
        
        # Volume momentum
        volume_momentum = indicators.get('volume_momentum', 0)
        if volume_momentum > 0.2:
            score += 0.4
        elif volume_momentum < -0.2:
            score -= 0.2
        factors += 1
        
        # Volume-price correlation
        volume_corr = indicators.get('volume_price_corr', 0)
        if volume_corr > 0.3:
            score += 0.3  # Positive correlation (healthy)
        elif volume_corr < -0.3:
            score -= 0.3  # Negative correlation (suspicious)
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _score_market_structure(self, indicators: Dict[str, float]) -> float:
        score = 0
        factors = 0
        
        # HH/LL
        market_structure = indicators.get('market_structure', 0)
        score += market_structure * 0.5
        factors += 1
        
        # Trend strength
        trend_strength = indicators.get('trend_strength', 0)
        score += np.sign(trend_strength) * min(abs(trend_strength)/2, 0.5)
        factors += 1
        
        # **ДОБАВИТЬ: R² — consistency**
        consistency = indicators.get('trend_consistency', 0)
        score += consistency * 0.3  # Чем выше R² → надёжнее тренд
        factors += 1
        
        # Breakout
        breakout = indicators.get('breakout_potential', 0)
        score += breakout * 0.4
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _score_momentum(self, indicators: Dict[str, float]) -> float:
        score = 0
        factors = 0
        
        # RSI analysis - более плавная шкала
        rsi_val = indicators.get('rsi', 50)
        if rsi_val < 35:  # Более мягкие границы
            score += 0.6
        elif rsi_val < 45:
            score += 0.3
        elif rsi_val > 65:
            score -= 0.6
        elif rsi_val > 55:
            score -= 0.3
        factors += 1
        
        # Price momentum - градуированная оценка
        mom_5d = indicators.get('momentum_5d', 0)
        mom_10d = indicators.get('momentum_10d', 0)
        
        if mom_5d > 0.05:
            score += 0.8
        elif mom_5d > 0.02:
            score += 0.4
        elif mom_5d < -0.05:
            score -= 0.8
        elif mom_5d < -0.02:
            score -= 0.4
        
        # Учитываем согласованность краткосрочного и среднесрочного импульса
        if mom_5d * mom_10d > 0:  # Одинаковое направление
            score += 0.2
        factors += 1

        # REMOVE OR FIX THE PROBLEMATIC SECTION:
        # vol_5d = indicators['close'].pct_change().tail(5).std()  # THIS CAUSES THE ERROR
        # if vol_5d > 0:
        #     risk_adj_mom = mom_5d / vol_5d
        # else:
        #     risk_adj_mom = 0
        
        # REPLACE WITH:
        # Use volatility from indicators instead of trying to calculate from 'close'
        vol_20d = indicators.get('volatility_20d', 0.02)
        if vol_20d > 0:
            risk_adj_mom = mom_5d / vol_20d
        else:
            risk_adj_mom = 0
        
        score += np.sign(risk_adj_mom) * min(abs(risk_adj_mom)*10, 0.8)
        factors += 1
        
        # Stochastic
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 25:  # Более мягкие границы
            score += 0.3
        elif stoch_k > 75:
            score -= 0.3
        factors += 1
        
        return score / max(factors, 1)
    
    def _score_volatility(self, indicators: Dict[str, float]) -> float:
        """Score volatility conditions"""
        score = 0
        factors = 0
        
        # Volatility level
        volatility = indicators.get('volatility_20d', 0.02)
        if 0.015 < volatility < 0.06:  # Optimal volatility range
            score += 0.5
        elif volatility > 0.1:  # Too high volatility
            score -= 0.3
        factors += 1
        
        # Bollinger Band position
        bb_pos = indicators.get('bb_position', 0.5)
        if bb_pos < 0.2:  # Near lower band
            score += 0.4
        elif bb_pos > 0.8:  # Near upper band
            score -= 0.4
        factors += 1
        
        # ATR percentage
        atr_pct = indicators.get('atr_percent', 0.02)
        if atr_pct < 0.015:  # Low volatility (potential breakout)
            score += 0.2
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _score_market_sentiment(self, indicators: Dict[str, float], symbol: str) -> float:
        """Score market sentiment using technicals as proxy"""
        score = 0
        factors = 0
        
        # Use RSI as sentiment proxy
        rsi_val = indicators.get('rsi', 50)
        if rsi_val > 60:
            score += 0.3  # Bullish sentiment
        elif rsi_val < 40:
            score -= 0.3  # Bearish sentiment
        factors += 1
        
        # Use volume as sentiment proxy
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            score += 0.2  # High interest
        factors += 1
        
        # Symbol-specific sentiment adjustments
        if symbol in ["BTC/USDT", "ETH/USDT"]:
            score += 0.1  # Slight bias toward majors
        factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _calculate_final_score_v2(self, scores: Dict[str, float]) -> float:
        total = 0
        weight = 0
        regime = scores['market_regime']
    
        # В консолидации — подавляем сильные сигналы
        if regime < 0.4:  # consolidation
            for cat in ['momentum', 'market_structure']:
                scores[cat] *= 0.5  # ослабляем
        elif regime > 0.7:  # strong trend
            scores['htf_alignment'] *= 1.3  # усиливаем
        # Старые веса
        for cat, sc in scores.items():
            if cat in self.config['weights']:
                w = self.config['weights'][cat]
                total += sc * w
                weight += w
        
        # Новые веса
        for cat, sc in scores.items():
            if cat in self.config['weights_v2']:
                w = self.config['weights_v2'][cat]
                total += sc * w
                weight += w
        
        
            
        return total / weight if weight > 0 else 0
    
    def _calculate_probability(self, score: float) -> float:
        """Convert score to probability"""
        k = self.config['logistic_k']
        x0 = self.config['logistic_x0']
        return 1.0 / (1.0 + math.exp(-k * (score - x0)))
    
    def _calculate_confidence(self, indicators: Dict[str, float], scores: Dict[str, float]) -> float:
        """Calculate confidence based on indicator strength and agreement"""
        confidence_factors = []
        
        # RSI confidence (extreme readings are more confident)
        rsi_val = indicators.get('rsi', 50)
        rsi_confidence = 1 - abs(rsi_val - 50) / 50  # Higher when RSI is extreme
        confidence_factors.append(rsi_confidence)
        
        # Volume confidence
        volume_ratio = indicators.get('volume_ratio', 1.0)
        volume_confidence = min(abs(volume_ratio - 1) * 2, 1.0)  # Higher with volume spikes
        confidence_factors.append(volume_confidence)
        
        # Trend strength confidence
        trend_strength = abs(indicators.get('trend_strength', 0))
        trend_confidence = min(trend_strength * 2, 1.0)
        confidence_factors.append(trend_confidence)
        
        # Score magnitude confidence
        final_score = self._calculate_final_score_v2(scores)
        score_confidence = abs(final_score)
        confidence_factors.append(score_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

# --------------------------------------------------
# DATA MANAGER
# --------------------------------------------------

class DataManager:
    """Manage data fetching and caching"""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        self.cache = {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
            all_ohlcv = []
            limit = 1000

            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                logger.info(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
                if len(ohlcv) > 0:
                    logger.info(f"First candle example: {ohlcv[0]}")  # Для дебага
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                if len(ohlcv) < limit:
                    break
                since = ohlcv[-1][0] + 1

            if not all_ohlcv:
                logger.warning(f"No OHLCV data for {symbol} {timeframe}")
                df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # Empty with columns
            else:
                try:
                    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                except ValueError as ve:
                    logger.error(f"Bad data format for {symbol} {timeframe}: {ve}")
                    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

            self.cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Fetch error {symbol} {timeframe}: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # Empty with columns

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def run_ultra_robust_analysis():
    """Run ultra-robust analysis without external APIs"""
    logger.info("Starting Ultra-Robust Narrative Analysis...")
    
    data_manager = DataManager(CONFIG['exchange'])
    scoring_engine = IntelligentScoringEngine(CONFIG)
    
    results = []
    
    for symbol in CONFIG['symbols']:
        logger.info(f"Analyzing {symbol}...")
        
        try:
            # Fetch OHLCV data
            df = data_manager.fetch_ohlcv(symbol, CONFIG['timeframe'], CONFIG['lookback_days'])
            
            if df.empty or len(df) < CONFIG['min_ohlcv_points']:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Analyze symbol
            result = scoring_engine.analyze_symbol(df, symbol, data_manager)
            if result:
                results.append(result)
                logger.info(f"Completed {symbol}: Score={result['final_score']:.3f}, "
                           f"ProbLong={result['prob_long']:.3f}, "
                           f"Confidence={result['confidence']:.3f}")
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue
    
    # Save and display results
    if results:
        save_ultra_robust_results(results)
        display_ultra_robust_summary(results)
    else:
        logger.warning("No results generated")
        
        try:
            logger.info("Выполняется анализ силы альткоинов...")
            alt_coins = [
                "bitcoin", "ethereum", "solana", "cardano",
                "tron", "polkadot", "chainlink", "litecoin", "dogecoin",
                "trump", "aster"
            ]

            alt_results = run_alt_analysis_coin_gecko(
                coin_ids=alt_coins,
                days=CONFIG.get("lookback_days", 90),
                top_n_for_others=10
            )

            if alt_results["indices"].empty:
                logger.warning("⚠️ Altcoin index data is empty. Skipping integration.")
            else:
                # Сохраняем в общий словарь
                try:
                    logger.info("Выполняется анализ силы альткоинов...")

                    alt_coins = [
                        "bitcoin", "ethereum", "solana", "cardano",
                        "tron", "polkadot", "chainlink", "litecoin", "dogecoin",
                        "trump", "aster"
                    ]

                    alt_results = run_alt_analysis_coin_gecko(
                        coin_ids=alt_coins,
                        days=CONFIG.get("lookback_days", 90),
                        top_n_for_others=10
                    )

                    if alt_results["indices"].empty:
                        logger.warning("⚠️ Altcoin index data is empty. Skipping integration.")
                    else:
                        # Преобразуем Timestamp в строки для JSON
                        indices_serializable = {
                            str(ts): {k: (float(v) if pd.notna(v) else None)
                                    for k, v in row.items()}
                            for ts, row in alt_results["indices"].tail(5).iterrows()
                        }

                        alt_strength_serializable = {
                            coin: {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                for k, v in data.items()}
                            for coin, data in alt_results["per_coin"].head(10).to_dict(orient="index").items()
                        }

                        alt_index_stats_serializable = {
                            k: (float(v) if isinstance(v, (int, float)) else str(v))
                            for k, v in alt_results["index_stats"].items()
                        }

                        results_dict = {
                            "assets": results,
                            "alt_indices": indices_serializable,
                            "alt_index_stats": alt_index_stats_serializable,
                            "alt_strength": alt_strength_serializable,
                        }

                        # Сохраняем JSON
                        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        combined_path = os.path.join(CONFIG["save_folder"], f"narrative_combined_{ts}.json")

                        with open(combined_path, "w", encoding="utf-8") as f:
                            json.dump(results_dict, f, indent=2, ensure_ascii=False)

                        logger.info(f"✅ Полный анализ (включая силу альтов) сохранён: {combined_path}")

                except Exception as e:
                    logger.error(f"Ошибка при анализе силы альтов: {e}")
        except Exception as r:
            print(r)
    return results

def save_ultra_robust_results(results: List[Dict[str, Any]]):
    """Save results"""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_file = os.path.join(CONFIG['save_folder'], f'ultra_robust_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV
    csv_data = []
    for result in results:
        row = {
            'symbol': result['symbol'],
            'final_score': result['final_score'],
            'prob_long': result['prob_long'],
            'confidence': result['confidence'],
            'price': result['price'],
            'price_change_24h': result['price_change_24h'],
            'volume_24h': result['volume_24h'],
        }
        
        # Add category scores
        for category, score in result['category_scores'].items():
            row[f'score_{category}'] = score
        
        # Add key indicators
        indicators = result['indicators']
        row.update({
            'rsi': indicators.get('rsi', 50),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'trend_strength': indicators.get('trend_strength', 0),
            'momentum_5d': indicators.get('momentum_5d', 0),
        })
        
        csv_data.append(row)
    
    csv_file = os.path.join(CONFIG['save_folder'], f'ultra_robust_{timestamp}.csv')
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    
    logger.info(f"Results saved with timestamp {timestamp}")

def display_ultra_robust_summary(results: List[Dict[str, Any]]):
    """Display summary"""
    print("\n" + "="*80)
    print("ULTRA-ROBUST NARRATIVE ANALYSIS SUMMARY")
    print("="*80)
    print("Using only OHLCV data - No external API dependencies")
    print(f"Assets analyzed: {len(results)}")
    print("\nTOP RECOMMENDATIONS:")
    print("-" * 80)
    
    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    for result in sorted_results:
        symbol = result['symbol']
        prob_long = result['prob_long']
        confidence = result['confidence']
        direction = "LONG" if prob_long > 0.5 else "SHORT"
        conviction = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.5 else "LOW"
        
        price = result['price']
        price_change = result['price_change_24h']
        volume = result['volume_24h']
        
        indicators = result['indicators']
        rsi_val = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        print(f"{symbol:15} {direction:5} [{conviction:6} conviction]")
        print(f"                 Probability: {prob_long:.3f} | Confidence: {confidence:.3f}")
        print(f"                 Price: ${price:8.2f} | 24h Change: {price_change:6.2f}%")
        print(f"                 RSI: {rsi_val:5.1f} | Volume Ratio: {volume_ratio:5.2f}x")
        
        # Top factors
        top_categories = sorted(result['category_scores'].items(), 
                              key=lambda x: abs(x[1]), reverse=True)[:2]
        factors_str = " | ".join([f"{cat}: {score:.3f}" for cat, score in top_categories])
        print(f"                 Key factors: {factors_str}")
        print()

# --------------------------------------------------
# TECHNICAL INDICATOR FUNCTIONS
# --------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=1).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[float, float]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(3).mean()
    
    return k.iloc[-1] if len(k) > 0 else 50, d.iloc[-1] if len(d) > 0 else 50

def calculate_atr_df(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR calculation"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr

def bollinger_bands(series: pd.Series, length: int = 20, mult: float = 2.0):
    """Bollinger Bands"""
    basis = sma(series, length)
    dev = mult * series.rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    return upper, basis, lower

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume - фиксируем dtype"""
    obv = pd.Series(0.0, index=close.index, dtype='float64')  # ← float64!
    obv.iloc[0] = float(volume.iloc[0])
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + float(volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - float(volume.iloc[i])
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
class AltMarketAnalyzer:
    """
    Fetch market caps & prices from CoinGecko, compute TOTAL/TOTAL2/TOTAL3/OTHERS.D,
    compute correlations and per-coin relative strength scores.
    """

    def __init__(self, vs_currency: str = "usd", max_retries: int = 3, session: Optional[requests.Session] = None):
        self.vs_currency = vs_currency
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "AltMarketAnalyzer/1.1"})

    def _fetch_with_retry(self, url: str, params: dict) -> Optional[dict]:
        """HTTP fetch with retry and exponential backoff for 429 errors."""
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=20)
                if r.status_code == 429:
                    logger.warning(f"⚠️ Rate limit hit ({r.status_code}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(f"Fetch failed ({attempt+1}/{self.max_retries}) for {url}: {e}")
                time.sleep(delay)
                delay *= 2
        logger.error(f"❌ Failed to fetch after {self.max_retries} attempts: {url}")
        return None

    def fetch_market_data(self, ids: List[str], days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch market_chart for multiple coins. Returns dict coin_id -> DataFrame.
        Handles retries and skips missing data.
        """
        out = {}
        for coin_id in ids:
            url = f"{COINGECKO_API_BASE}/coins/{coin_id}/market_chart"
            params = {"vs_currency": self.vs_currency, "days": days, "interval": "daily"}
            data = load_from_cache(coin_id, days)
            if not data:
                data = self._fetch_with_retry(url, params)
                if data:
                    save_to_cache(coin_id, days, data)
            if not data or "prices" not in data:
                logger.warning(f"CoinGecko data missing for {coin_id}")
                continue

            try:
                df = pd.DataFrame({
                    "timestamp": [pd.to_datetime(p[0], unit="ms").normalize() for p in data["prices"]],
                    "price": [p[1] for p in data["prices"]],
                    "market_cap": [m[1] for m in data["market_caps"]],
                    "volume": [v[1] for v in data["total_volumes"]],
                }).dropna()
                df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
                if not df.empty:
                    out[coin_id] = df
            except Exception as e:
                logger.warning(f"Failed to parse {coin_id}: {e}")
        return out

    def compute_indices(self, coin_dfs: Dict[str, pd.DataFrame], top_n_for_others: int = 10) -> pd.DataFrame:
        """Build TOTAL, TOTAL2, TOTAL3, OTHERS.D indices safely."""
        if not coin_dfs:
            logger.warning("⚠️ No coin data available for index computation.")
            return pd.DataFrame()

        all_dates = sorted({d for df in coin_dfs.values() for d in df.index})
        if not all_dates:
            logger.warning("⚠️ No valid timestamps found in coin data.")
            return pd.DataFrame()

        mcap_matrix = pd.DataFrame(index=all_dates)
        latest_caps = {}
        for cid, df in coin_dfs.items():
            mcap_series = df["market_cap"].reindex(all_dates).fillna(method="ffill")
            mcap_matrix[cid] = mcap_series
            latest_caps[cid] = mcap_series.dropna().iloc[-1] if not mcap_series.dropna().empty else 0

        if mcap_matrix.empty:
            logger.warning("⚠️ Market cap matrix is empty.")
            return pd.DataFrame()

        total = mcap_matrix.sum(axis=1)
        btc_id = "bitcoin" if "bitcoin" in mcap_matrix.columns else None
        eth_id = "ethereum" if "ethereum" in mcap_matrix.columns else None

        total2 = total - (mcap_matrix[btc_id] if btc_id else 0)
        total3 = total2 - (mcap_matrix[eth_id] if eth_id else 0)

        top_n = sorted(latest_caps.items(), key=lambda x: x[1], reverse=True)[:top_n_for_others]
        top_ids = [x[0] for x in top_n]
        others_mcap = mcap_matrix.drop(columns=top_ids, errors="ignore").sum(axis=1)
        others_d = others_mcap / total.replace(0, np.nan)

        df_indices = pd.DataFrame({
            "TOTAL": total,
            "TOTAL2": total2,
            "TOTAL3": total3,
            "OTHERS_MCAP": others_mcap,
            "OTHERS.D": others_d,
        }).dropna()

        return df_indices

    def compute_index_stats(self, df_indices: pd.DataFrame) -> Dict[str, Any]:
        """Compute pct changes and correlations, with safety checks."""
        stats = {}
        if df_indices.empty:
            logger.warning("⚠️ Empty index DataFrame — skipping stats.")
            return stats

        for col in ["TOTAL", "TOTAL2", "TOTAL3", "OTHERS.D"]:
            if col not in df_indices or len(df_indices[col].dropna()) < 2:
                stats[f"{col}_pct_1d"] = np.nan
                stats[f"{col}_pct_7d"] = np.nan
                stats[f"{col}_pct_30d"] = np.nan
                continue
            stats[f"{col}_pct_1d"] = df_indices[col].pct_change(1).iloc[-1]
            stats[f"{col}_pct_7d"] = df_indices[col].pct_change(min(7, len(df_indices) - 1)).iloc[-1]
            stats[f"{col}_pct_30d"] = df_indices[col].pct_change(min(30, len(df_indices) - 1)).iloc[-1]

        try:
            stats["last_corr_matrix"] = df_indices[["TOTAL", "TOTAL2", "TOTAL3"]].pct_change().corr()
        except Exception:
            stats["last_corr_matrix"] = pd.DataFrame()
        return stats

    def compute_alt_strength_per_coin(self, coin_dfs: Dict[str, pd.DataFrame], days_short=7, days_long=30) -> pd.DataFrame:
        """Compute per-coin relative strength (return, vol, vol-adjusted)."""
        if not coin_dfs:
            return pd.DataFrame()

        rows = []
        btc_series = coin_dfs.get("bitcoin", pd.DataFrame()).get("price") if "bitcoin" in coin_dfs else None
        eth_series = coin_dfs.get("ethereum", pd.DataFrame()).get("price") if "ethereum" in coin_dfs else None

        for cid, df in coin_dfs.items():
            if df.empty:
                continue
            price = df["price"].dropna()
            if len(price) < 3:
                continue
            try:
                r7 = price.pct_change(days_short).iloc[-1] if len(price) > days_short else np.nan
                r30 = price.pct_change(days_long).iloc[-1] if len(price) > days_long else np.nan
                vol7 = price.pct_change().rolling(days_short).std().iloc[-1]
                vol30 = price.pct_change().rolling(days_long).std().iloc[-1]
                mcap = df["market_cap"].iloc[-1]

                rel_vs_btc_7 = (r7 / btc_series.pct_change(days_short).iloc[-1]) if btc_series is not None else np.nan
                rel_vs_eth_7 = (r7 / eth_series.pct_change(days_short).iloc[-1]) if eth_series is not None else np.nan
                vol_adj_7 = r7 / (vol7 + 1e-12)
                rows.append({
                    "coin": cid,
                    "r7": r7,
                    "r30": r30,
                    "vol7": vol7,
                    "vol30": vol30,
                    "mcap": mcap,
                    "rel_vs_btc_7": rel_vs_btc_7,
                    "rel_vs_eth_7": rel_vs_eth_7,
                    "vol_adj_7": vol_adj_7,
                })
            except Exception as e:
                logger.warning(f"Error computing alt strength for {cid}: {e}")

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows).set_index("coin")
        df_out["vol_adj_7_z"] = (df_out["vol_adj_7"] - df_out["vol_adj_7"].mean()) / (df_out["vol_adj_7"].std() + 1e-12)
        df_out["rel_strength_score"] = (
            df_out["r7"].rank(pct=True) * 0.5
            + ((df_out["vol_adj_7_z"] - df_out["vol_adj_7_z"].min()) /
               (df_out["vol_adj_7_z"].max() - df_out["vol_adj_7_z"].min() + 1e-12) * 0.5)
        )
        return df_out.sort_values("rel_strength_score", ascending=False)


def run_alt_analysis_coin_gecko(coin_ids: List[str], days: int = 90, top_n_for_others: int = 10):
    """Wrapper with full fault-tolerance and safe CSV saving."""
    ama = AltMarketAnalyzer()
    coin_dfs = ama.fetch_market_data(coin_ids, days=days)
    if not coin_dfs:
        logger.error("No data fetched from CoinGecko — returning empty result.")
        return {"indices": pd.DataFrame(), "index_stats": {}, "per_coin": pd.DataFrame()}

    df_indices = ama.compute_indices(coin_dfs, top_n_for_others=top_n_for_others)
    idx_stats = ama.compute_index_stats(df_indices)
    per_coin = ama.compute_alt_strength_per_coin(coin_dfs)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_dir = CONFIG.get("save_folder", "./")
    os.makedirs(save_dir, exist_ok=True)
    if not df_indices.empty:
        df_indices.to_csv(os.path.join(save_dir, f"alt_indices_{ts}.csv"))
    if not per_coin.empty:
        per_coin.to_csv(os.path.join(save_dir, f"alt_strength_{ts}.csv"))

    return {"indices": df_indices, "index_stats": idx_stats, "per_coin": per_coin}

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
coins = [
        "bitcoin", "ethereum", "solana", "cardano", "tron", 
        "polkadot", "chainlink", "litecoin", "dogecoin", "avalanche-2",
        "trump", 'aster'
    ]
if __name__ == '__main__':
    print("Ultra-Robust Narrative Analyzer starting...")
    print("No external API dependencies - Using advanced technical analysis only")

    results = run_ultra_robust_analysis()

    if results:
        print(f"\nAnalysis complete! Check {CONFIG['save_folder']} for detailed results.")
    else:
        print("\nAnalysis completed but no results were generated.")