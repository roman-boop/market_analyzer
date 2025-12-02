## Ultra-Robust Narrative Analyzer
Вероятностная MTF-модель анализа финансовых рынков на основе чистого OHLCV

Канал автора по финансовым рынкам, ручной торговле и алготрейдингу https://t.me/crypto_maniacdt

Этот проект — исследовательский движок вероятностного анализа рынка, построенный на:

1. многовременном анализе (MTF)
2. регрессиях тренда (slope + R²)
3. оценке волатильности (ATR, historical volatility)
4. индикаторах momentum / volume / market structure
5. рыночных режимах (trend / consolidation / mixed)
6. логистической модели вероятности
7. анализе TOTAL / TOTAL2 / TOTAL3 / OTHERS.D с CoinGecko

   Система превращает технические и статистические признаки в итоговый final_score,
который затем нормализуется в prob_long — вероятность роста актива.

## Основные возможности:

1. Загрузка OHLCV через CCXT
2. Индикаторы: RSI, Stoch, ATR, ADX, OBV, ROC, BB, Volume Trends
3. Market Structure Engine
4. Volatility Engine
5. Momentum Engine
6. Multi-Timeframe Analyzer (LTF / Daily / HTF)
7. Volatility Regime + Market Regime
8. Confidence Model
9. Alt-market анализ (TOTAL, TOTAL2, TOTAL3, OTHERS.D)
10. Финальная вероятностная модель ProbLong
11. Экспорт в JSON/CSV и текстовый отчёт

Подходит как база для алготрейдинга / Telegram-ботов / аналитики
