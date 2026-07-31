# -*- coding: utf-8 -*-
# ============================================================
# SVR 估值多空— 修复版
#
# 修复点：
#   1. 每次调仓都重新训练 SVR（原版只在首次调仓训练一次，此后十年沿用
#      首月截面拟合的模型与 scaler——横截面相对估值模型必须逐期重估）
#   2. select 阶段统一用 ndarray 输入 scaler.transform，消除 sklearn
#      特征名不一致告警
#   3. 其余逻辑（多空预算 0.60/0.35、A股做空不支持时自动降级为纯多头、
#      决策日用 previous_date 防未来函数）原版已正确，保持不变
# ============================================================

from jqdata import *
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ========= Parameters =========
TARGET_INDEX = '000300.XSHG'
BENCHMARK_INDEX = TARGET_INDEX

UNIVERSE_N = 100
MAX_LONG = 15
MAX_SHORT = 15

MIN_TRAIN_SAMPLES = 20
POSITION_RATIO = 0.95

LONG_RATIO = 0.60
SHORT_RATIO = 0.35

ENABLE_SHORT = True
MIN_ORDER_CASH = 1000
# ==============================


def initialize(context):
    set_benchmark(BENCHMARK_INDEX)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    cost = OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    )
    set_order_cost(cost, type='stock')
    set_slippage(FixedSlippage(0.002))

    g.last_rebalance = None
    g.scaler = None
    g.svr_model = None
    g.short_supported = True

    run_monthly(trade, 1, time='9:40', reference_security=BENCHMARK_INDEX)
    run_daily(record_data, time='after_close')


def signlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def get_stock_pool(decision_date):
    try:
        stocks = get_index_stocks(TARGET_INDEX, date=decision_date)
        stocks = sorted(stocks)
        return stocks[:UNIVERSE_N]
    except Exception as e:
        log.info('Error getting stock pool: {}'.format(e))
        return []


def fetch_fundamentals(stocks, decision_date):
    if not stocks:
        return None
    try:
        q = query(
            valuation.code,
            valuation.market_cap,
            valuation.pe_ratio,
            valuation.pb_ratio,
            indicator.roe,
            indicator.roa,
            indicator.gross_profit_margin,
            indicator.net_profit_margin,
            indicator.inc_total_revenue_year_on_year,
            indicator.inc_net_profit_year_on_year
        ).filter(valuation.code.in_(stocks))

        df = get_fundamentals(q, date=decision_date)
        if df is None or df.empty:
            log.info('No fundamentals for decision_date={}'.format(decision_date))
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


def prepare_features(df_raw):
    if df_raw is None or df_raw.empty:
        return None

    df = df_raw.copy()
    if 'market_cap' not in df.columns:
        return None

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median()).fillna(0)

    df['market_cap'] = np.log1p(np.maximum(df['market_cap'].values, 0))

    for col in df.columns:
        if col != 'market_cap':
            df[col] = signlog(df[col].values)

    df = df.select_dtypes(include=[np.number])

    if 'market_cap' not in df.columns:
        return None

    df = df.astype(np.float64)
    return df


def train_svr(df_feat):
    """每次调仓重训（修复1）：横截面相对估值需逐期重估"""
    if df_feat is None or df_feat.empty or len(df_feat) < MIN_TRAIN_SAMPLES:
        log.info('Sample size too small: {}'.format(0 if df_feat is None else len(df_feat)))
        return False

    try:
        y = pd.to_numeric(df_feat['market_cap'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        x = df_feat.drop('market_cap', axis=1).copy()
        x = x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

        X = np.nan_to_num(np.asarray(x.values, dtype=np.float64))
        Y = np.nan_to_num(np.asarray(y.values, dtype=np.float64))

        if X.shape[1] < 3:
            log.info('Not enough features: {}'.format(X.shape[1]))
            return False

        g.scaler = StandardScaler()
        Xs = g.scaler.fit_transform(X)

        g.svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        g.svr_model.fit(Xs, Y)

        log.info('SVR trained successfully')
        return True

    except Exception as e:
        log.info('Error training SVR: {}'.format(e))
        g.svr_model = None
        return False


def select_long_short_by_svr(df_feat):
    try:
        x = df_feat.drop('market_cap', axis=1).copy()
        x = x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        X = np.asarray(x.values, dtype=np.float64)   # 修复2：ndarray 输入

        Xs = g.scaler.transform(X)
        pred = g.svr_model.predict(Xs)
        actual = np.asarray(df_feat['market_cap'].values, dtype=np.float64)

        score = pred - actual
        pairs = list(zip(df_feat.index.tolist(), score))
        pairs.sort(key=lambda t: t[1], reverse=True)

        long_list = [s for s, _ in pairs[:MAX_LONG]]
        short_list = [s for s, _ in pairs[-MAX_SHORT:]]
        short_list = [s for s in short_list if s not in set(long_list)]
        return long_list, short_list

    except Exception as e:
        log.info('Error selecting by SVR: {}'.format(e))
        return [], []


def select_long_short_by_value(df_raw):
    """
    SVR fails fallback:
    value_score = 0.5*z(roe) + 0.25*z(1/pe) + 0.25*z(1/pb)
    """
    df = df_raw.copy()

    for col in ['pe_ratio', 'pb_ratio', 'roe']:
        if col not in df.columns:
            df[col] = 0

    for col in ['pe_ratio', 'pb_ratio', 'roe']:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

    df['pe_ratio'] = np.clip(df['pe_ratio'], 1, 200)
    df['pb_ratio'] = np.clip(df['pb_ratio'], 0.5, 50)

    df['inv_pe'] = 1.0 / df['pe_ratio']
    df['inv_pb'] = 1.0 / df['pb_ratio']

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 1e-12 else 0

    df['score'] = 0.5 * zscore(df['roe']) + 0.25 * zscore(df['inv_pe']) + 0.25 * zscore(df['inv_pb'])

    df = df.sort_values('score', ascending=False)
    long_list = df.head(MAX_LONG).index.tolist()

    df2 = df.sort_values('score', ascending=True)
    short_list = df2.head(MAX_SHORT).index.tolist()
    short_list = [s for s in short_list if s not in set(long_list)]
    return long_list, short_list


def get_close_price(stock, decision_date):
    px = get_price(stock, end_date=decision_date, count=1, fields=['close'])
    if px is None or px.empty:
        return None
    p = float(px['close'].iloc[0])
    if p <= 0:
        return None
    return p


def execute_rebalance_long_short(context, decision_date, long_list, short_list):
    total_value = context.portfolio.total_value
    invest_value = total_value * POSITION_RATIO

    if (not ENABLE_SHORT) or (not g.short_supported) or (not short_list):
        long_budget = invest_value
        short_budget = 0.0
        short_list = []
    else:
        long_budget = invest_value * LONG_RATIO
        short_budget = invest_value * SHORT_RATIO

    cand = []
    for s in long_list:
        p = get_close_price(s, decision_date)
        if p is None:
            continue
        cand.append((s, p))

    cand.sort(key=lambda t: t[1])  # cheaper first

    chosen = []
    if cand and long_budget >= MIN_ORDER_CASH:
        max_n = min(MAX_LONG, len(cand))
        for n in range(max_n, 0, -1):
            each_value = long_budget / n
            max_price = max([p for _, p in cand[:n]])
            if each_value >= 100 * max_price:
                chosen = cand[:n]
                break

    keep = set([s for s, _ in chosen])
    for s in list(context.portfolio.positions.keys()):
        if s not in keep:
            order_target(s, 0)

    long_placed = 0
    if chosen:
        each_value = long_budget / len(chosen)
        for s, price in chosen:
            if each_value < MIN_ORDER_CASH:
                continue
            target_shares = int(each_value / price / 100) * 100
            if target_shares < 100:
                continue

            cur = context.portfolio.positions[s].total_amount if s in context.portfolio.positions else 0
            delta = target_shares - cur
            if delta == 0:
                continue

            r = order(s, delta)
            if r is not None:
                long_placed += 1

    short_placed = 0
    if ENABLE_SHORT and g.short_supported and short_list and short_budget >= MIN_ORDER_CASH:
        max_s = min(MAX_SHORT, len(short_list))
        each_value_s = short_budget / max(1, max_s)

        for s in short_list[:max_s]:
            p = get_close_price(s, decision_date)
            if p is None:
                continue
            target_shares = int(each_value_s / p / 100) * 100
            if target_shares < 100:
                continue
            try:
                r = order(s, -target_shares)
                if r is not None:
                    short_placed += 1
            except Exception as e:
                log.info('Short not supported, disable. err={}'.format(e))
                g.short_supported = False
                break

    log.info('chosen long count: {}'.format(len(chosen)))
    log.info('placed orders: long={}, short={}, short_supported={}'.format(long_placed, short_placed, g.short_supported))


def trade(context):
    today = context.current_dt.date()
    decision_date = context.previous_date

    log.info('===== rebalance {} (decision_date={}) ====='.format(today, decision_date))
    log.info('portfolio value: {:.2f}'.format(context.portfolio.total_value))

    if g.last_rebalance and (today - g.last_rebalance).days < 20:
        return

    stocks = get_stock_pool(decision_date)
    log.info('universe size: {}'.format(len(stocks)))
    if len(stocks) < 10:
        log.info('universe too small, skip')
        return

    df_raw = fetch_fundamentals(stocks, decision_date)
    if df_raw is None or df_raw.empty:
        log.info('no fundamentals, skip')
        return

    df_feat = prepare_features(df_raw)
    if df_feat is None or df_feat.empty:
        log.info('feature prep failed, skip')
        return
    log.info('df_feat shape: {}'.format(df_feat.shape))

    # 修复1：每次调仓都用当期截面重训模型
    ok = train_svr(df_feat)
    log.info('SVR trained: {}'.format(ok))

    long_list, short_list = [], []
    if ok and g.svr_model is not None and g.scaler is not None:
        long_list, short_list = select_long_short_by_svr(df_feat)

    if (not long_list) or len(long_list) < max(3, MAX_LONG // 2):
        log.info('SVR long targets insufficient, fallback to value score long-short')
        long_list, short_list = select_long_short_by_value(df_raw)

    log.info('long targets ({}): {}'.format(len(long_list), long_list[:10]))
    log.info('short targets ({}): {}'.format(len(short_list), short_list[:10]))

    execute_rebalance_long_short(context, decision_date, long_list, short_list)

    g.last_rebalance = today


def record_data(context):
    record(
        portfolio_value=context.portfolio.total_value,
        hold_num=len(context.portfolio.positions),
        cash=context.portfolio.cash,
        short_supported=int(g.short_supported)
    )

