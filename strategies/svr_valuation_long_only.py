# -*- coding: utf-8 -*-

from jqdata import *
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ========= Parameters =========
MAX_HOLD = 15
UNIVERSE_N = 100
MIN_TRAIN_SAMPLES = 20
POSITION_RATIO = 0.95
# ==============================

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')

    set_slippage(FixedSlippage(0.002))

    g.last_rebalance = None
    g.scaler = None
    g.svr_model = None
    g.model_trained = False

    # Rebalance monthly at 09:40 (trade today, but decide using previous_date fundamentals)
    run_monthly(trade, 1, time='9:40', reference_security='000300.XSHG')
    run_daily(record_data, time='after_close')


def signlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def get_stock_pool(decision_date):
    """
    Use index constituents as universe.
    IMPORTANT: use decision_date (previous trading day) to avoid future data issues.
    """
    try:
        stocks = get_index_stocks('000300.XSHG', date=decision_date)
        if len(stocks) < UNIVERSE_N:
            extra = get_index_stocks('000905.XSHG', date=decision_date)
            stocks = list(set(stocks + extra))
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

        # 强制转成数值（把字符串/None 都变成 NaN）
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)

        # 先用中位数填，再用 0 填（避免全 NaN 的列）
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

    # 再次强制数值（双保险）
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median()).fillna(0)

    # target transform
    df['market_cap'] = np.log1p(np.maximum(df['market_cap'].values, 0))

    # feature transform
    for col in df.columns:
        if col != 'market_cap':
            df[col] = signlog(df[col].values)

    # 最关键：只保留数值列（彻底消灭 str/object）
    df = df.select_dtypes(include=[np.number])

    # 再确认一次 target 存在
    if 'market_cap' not in df.columns:
        return None

    return df




def train_svr(df):
    if df is None or df.empty or len(df) < MIN_TRAIN_SAMPLES:
        log.info('Sample size too small: {}'.format(0 if df is None else len(df)))
        return False

    try:
        y = df['market_cap'].astype(float).copy()
        x = df.drop('market_cap', axis=1)

        # 强制 float（再保险一次）
        x = x.astype(float)

        if x.shape[1] < 3:
            log.info('Not enough features: {}'.format(x.shape[1]))
            return False

        # 打印 dtype 检查信息：应当全是 float/int
        log.info('X dtypes summary: {}'.format(x.dtypes.value_counts().to_dict()))

        g.scaler = StandardScaler()
        x_scaled = g.scaler.fit_transform(x.values)

        g.svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', max_iter=500)
        g.svr_model.fit(x_scaled, y.values)

        # 简单训练质量检查：相关系数（不是必须，但很有用）
        pred = g.svr_model.predict(x_scaled)
        corr = np.corrcoef(pred, y.values)[0, 1] if len(y) > 1 else np.nan
        log.info('SVR trained successfully, corr={}'.format(corr))

        g.model_trained = True
        return True

    except Exception as e:
        log.info('Error training SVR: {}'.format(e))
        g.model_trained = False
        return False



def select_by_svr(df):
    """
    residual = predicted - actual
    choose largest residual => predicted > actual => undervalued
    """
    try:
        x = df.drop('market_cap', axis=1)
        x_scaled = g.scaler.transform(x)
        pred = g.svr_model.predict(x_scaled)
        actual = df['market_cap'].values

        score = pred - actual
        pairs = list(zip(df.index.tolist(), score))
        pairs.sort(key=lambda t: t[1], reverse=True)
        selected = [s for s, _ in pairs[:MAX_HOLD]]
        return selected

    except Exception as e:
        log.info('Error selecting by SVR: {}'.format(e))
        return []


def fallback_value_select(df_raw):
    """
    Simple fallback: high ROE + low PE/PB
    """
    try:
        df = df_raw.copy()

        for col in ['pe_ratio', 'pb_ratio', 'roe']:
            if col not in df.columns:
                df[col] = 0

        df['pe_ratio'] = pd.to_numeric(df['pe_ratio'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        df['pb_ratio'] = pd.to_numeric(df['pb_ratio'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        df['roe'] = pd.to_numeric(df['roe'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

        df['pe_ratio'] = np.clip(df['pe_ratio'], 1, 200)
        df['pb_ratio'] = np.clip(df['pb_ratio'], 0.5, 50)

        df['inv_pe'] = 1.0 / df['pe_ratio']
        df['inv_pb'] = 1.0 / df['pb_ratio']

        def zscore(s):
            std = s.std()
            return (s - s.mean()) / std if std > 1e-12 else 0

        df['score'] = 0.5 * zscore(df['roe']) + 0.25 * zscore(df['inv_pe']) + 0.25 * zscore(df['inv_pb'])
        df = df.sort_values('score', ascending=False)
        return df.head(MAX_HOLD).index.tolist()

    except Exception as e:
        log.info('Fallback select error: {}'.format(e))
        return []


def execute_rebalance(context, targets):
    if not targets:
        return

    current_positions = list(context.portfolio.positions.keys())

    # sell
    for s in current_positions:
        if s not in targets:
            order_target(s, 0)

    # buy
    cdata = get_current_data()
    cash = context.portfolio.cash * POSITION_RATIO
    each_value = cash / len(targets)

    placed = 0
    for s in targets:
        if cdata[s].paused:
            continue
        if each_value < 1000:
            continue
        r = order_value(s, each_value)
        if r is not None:
            placed += 1

    log.info('placed buy orders: {}'.format(placed))



def trade(context):
    today = context.current_dt.date()

    # IMPORTANT: decision_date uses previous trading day to avoid future data
    decision_date = context.previous_date

    log.info('===== rebalance {} (decision_date={}) ====='.format(today, decision_date))
    log.info('portfolio value: {:.2f}'.format(context.portfolio.total_value))

    # guard: too frequent rebalance
    if g.last_rebalance and (today - g.last_rebalance).days < 20:
        return

    # 1) universe
    stocks = get_stock_pool(decision_date)
    log.info('universe size: {}'.format(len(stocks)))
    if len(stocks) < 10:
        log.info('universe too small, skip')
        return

    # 2) fundamentals
    df_raw = fetch_fundamentals(stocks, decision_date)
    if df_raw is None or df_raw.empty:
        log.info('no fundamentals, skip')
        return

    # 3) features
    df_feat = prepare_features(df_raw)
    if df_feat is None or df_feat.empty:
        log.info('feature prep failed, skip')
        return
    log.info('df_feat shape: {}'.format(df_feat.shape))
    log.info('df_feat has object dtype? {}'.format(any(df_feat.dtypes == "object")))


    # 4) train model if needed
    if (not g.model_trained) or (g.svr_model is None):
        ok = train_svr(df_feat)
        log.info('SVR trained: {}'.format(ok))

    # 5) select
    targets = []
    if g.model_trained and g.svr_model is not None and g.scaler is not None:
        targets = select_by_svr(df_feat)

    if not targets or len(targets) < max(3, MAX_HOLD // 2):
        log.info('SVR targets insufficient, fallback to value select')
        targets = fallback_value_select(df_raw)

    if not targets:
        log.info('no targets, skip')
        return

    log.info('targets ({}): {}'.format(len(targets), targets[:10]))

    # 6) rebalance
    execute_rebalance(context, targets)
    g.last_rebalance = today


def record_data(context):
    record(
        portfolio_value=context.portfolio.total_value,
        hold_num=len(context.portfolio.positions),
        cash=context.portfolio.cash
    )

