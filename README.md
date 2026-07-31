# Quant Strategies (JoinQuant / A股)

A collection of quantitative trading strategies developed and backtested on the
[JoinQuant](https://www.joinquant.com) platform, covering fundamental factor
investing, machine-learning-based valuation, volatility timing and options
volatility-surface modeling.

基于聚宽（JoinQuant）平台开发与回测的 A 股量化策略集：基本面因子、机器学习估值、
波动率择时与期权波动率曲面建模。

## Strategies 策略一览

| Strategy | File | Idea | Backtest (JoinQuant) |
|---|---|---|---|
| **ROE+PE+PB 指数增强** | [`strategies/roe_pe_pb_index_enhanced.py`](strategies/roe_pe_pb_index_enhanced.py) | 中证500成分内按 `0.5·z(ROE) + 0.25·z(1/PE) + 0.25·z(1/PB)` 综合评分，月度调仓持有 Top 10 | **2010–2020（10 年）：+27.50% vs 基准 +17.44%**，最大回撤 30.18% |
| **SVR 估值选股（纯多头）** | [`strategies/svr_valuation_long_only.py`](strategies/svr_valuation_long_only.py) | SVR(rbf) 学习「基本面 → log 市值」映射，残差为估值偏离度，做多低估股票 | 迭代版本 |
| **SVR 估值多空** | [`strategies/svr_valuation_long_short.py`](strategies/svr_valuation_long_short.py) | 同上估值框架，多头 60% / 空头 35% 预算，A 股做空受限时自动降级纯多头 | 2012–2022（10 年）：+42.60% |
| **50ETF 波动率择时** | [`strategies/etf50_volatility_timing.py`](strategies/etf50_volatility_timing.py) | 20 日历史波动率的 60 日 z-score：低波动做多 / 高波动空仓 | 2016–2020：+19.65%，回撤 21.20%（防御型，牛市段跑输基准属策略特性） |
| **波动率曲面凸度套利（框架）** | [`strategies/volatility_surface_convexity_arbitrage.py`](strategies/volatility_surface_convexity_arbitrage.py) | 三维波动率曲面（期限×虚实度，三次插值）+ 凸度异常检测 + BS 定价 / 隐波求解 / Greeks 全套实现 | 演示框架（期权链数据为模拟生成） |

## Fixed versions 修复版

`fixed/` 目录包含对早期版本的工程修正，并附完整修正说明：

- [`fixed/roe_pe_pb_index_enhanced_fixed.py`](fixed/roe_pe_pb_index_enhanced_fixed.py)
  — 修复未来函数风险（决策一律基于 T-1 数据 + `avoid_future_data`）与调仓权重逻辑
  （改为组合总值等权 + `order_target_value`）
- [`fixed/svr_valuation_long_short_fixed.py`](fixed/svr_valuation_long_short_fixed.py)
  — 横截面估值模型改为每期重训（原版仅首期训练一次）

原版文件保留原始回测口径；修复后重跑结果与历史回测存在差异属正常现象——
消除未来函数后收益通常回落，这正是保留两版对照的意义。

## Notes

- Python 3 / JoinQuant API（`jqdata`）、scikit-learn、SciPy
- 所有回测均含交易成本与滑点设置（佣金 0.03%、卖出印花税 0.1%、滑点 0.2%）
- For research & educational purposes only. **Not investment advice.**
  仅供研究学习，不构成任何投资建议。

