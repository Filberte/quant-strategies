# -*- coding: utf-8 -*-
# ============================================================
# ROE+PE+PB 指数增强— 修复版
# 原版回测（2010-01-01 ~ 2020-01-01）：策略 27.50% vs 基准 17.44%，最大回撤 30.18%
# 注意：修复未来函数与调仓逻辑后重跑，结果与原版历史回测会有差异，属正常现象。
#
# 修复点：
#   1. set_option('avoid_future_data', True)：显式开启未来数据防护（原版缺失）
#   2. 决策数据一律使用 decision_date = context.previous_date（原版用当日数据，
#      在 time='open' 场景下存在用到当日收盘价/当日财务快照的未来函数风险）
#   3. 调仓改为按组合总值等权 + order_target_value（原版只按"当前现金"均分并手工
#      算股数，会导致保留持仓被错误减仓、目标权重失真、换手异常偏高）
#   4. 调仓时间从 time='open' 改为 9:40，避开开盘竞价异常价格
# ============================================================
from jqdata import *
import pandas as pd
import numpy as np

# ========= Params =========
INDEX_CODE = '000905.XSHG'   # CSI 500
MAX_HOLD = 10                # 最大持仓数
UNIVERSE_N = 50              # 按市值取前 N 只
POSITION_RATIO = 0.95        # 仓位比例
# =========================


def initialize(context):
    set_benchmark(INDEX_CODE)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)   # 修复1

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')
    set_slippage(FixedSlippage(0.002))

    run_monthly(rebalance, 1, time='9:40', reference_security=INDEX_CODE)  # 修复4
    run_daily(daily_record, time='after_close')

    g.rebalance_flag = 0


def fetch_fundamentals_simple(stocks, date_):
    """获取基本面数据（market_cap / pe / pb / roe）"""
    if not stocks:
        return None
    try:
        q = query(
            valuation.code,
            valuation.market_cap,
            valuation.pe_ratio,
            valuation.pb_ratio,
            indicator.roe
        ).filter(valuation.code.in_(stocks))

        df = get_fundamentals(q, date=date_)
        if df is None or df.empty:
            log.info('No fundamentals data for {}'.format(date_))
            return None

        df = df.set_index('code')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median()).fillna(0)
        return df

    except Exception as e:
        log.info('Error fetching fundamentals: {}'.format(e))
        return None


def rebalance(context):
    today = context.current_dt.date()
    decision_date = context.previous_date   # 修复2：决策一律基于前一交易日
    g.rebalance_flag = 1
    log.info('===== rebalance {} (decision_date={}) ====='.format(today, decision_date))
    log.info('portfolio value: {:.2f}'.format(context.portfolio.total_value))

    # 1) 股票池：中证500成分（按决策日）
    try:
        universe = get_index_stocks(INDEX_CODE, date=decision_date)
    except Exception as e:
        log.info('Error getting index stocks: {}'.format(e))
        universe = []
    if len(universe) < 10:
        log.info('universe too small, skip')
        return

    # 2) 基本面（按决策日）
    df = fetch_fundamentals_simple(universe, decision_date)
    if df is None or df.empty:
        log.info('no fundamentals, skip')
        return

    # 3) 按市值取前 UNIVERSE_N，计算 ROE+PE+PB 综合评分
    df = df.sort_values('market_cap', ascending=False).head(UNIVERSE_N)

    df['pe_ratio'] = np.clip(df['pe_ratio'], 1, 200)
    df['pb_ratio'] = np.clip(df['pb_ratio'], 0.5, 50)
    df['inv_pe'] = 1.0 / df['pe_ratio']
    df['inv_pb'] = 1.0 / df['pb_ratio']

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 1e-12 else s * 0.0

    df['score'] = (0.5 * zscore(df['roe'])
                   + 0.25 * zscore(df['inv_pe'])
                   + 0.25 * zscore(df['inv_pb']))
    df['score'] = df['score'].replace([np.inf, -np.inf], 0).fillna(0)

    target = df.sort_values('score', ascending=False).head(MAX_HOLD).index.tolist()

    # 4) 过滤停牌后执行调仓（修复3：总值等权 + order_target_value）
    cdata = get_current_data()
    target = [s for s in target if s in cdata and not cdata[s].paused]
    if not target:
        log.info('no tradable targets, skip')
        return
    log.info('targets ({}): {}'.format(len(target), target))

    # 先清掉非目标持仓
    for s in list(context.portfolio.positions.keys()):
        if s not in target:
            order_target(s, 0)

    # 再把每只目标调到等权目标市值
    each_value = context.portfolio.total_value * POSITION_RATIO / len(target)
    for s in target:
        order_target_value(s, each_value)

    log.info('rebalance completed')


def daily_record(context):
    record(
        portfolio_value=context.portfolio.total_value,
        hold_num=len(context.portfolio.positions),
        cash=context.portfolio.cash,
        rebalance_flag=g.rebalance_flag
    )
    g.rebalance_flag = 0

