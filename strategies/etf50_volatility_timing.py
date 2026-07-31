# -*- coding: utf-8 -*-
from jqdata import *
import numpy as np
import pandas as pd
import math

UNDERLYING = "510050.XSHG"
BENCHMARK = "000300.XSHG"

HV_WINDOW = 20
Z_WINDOW = 60

POSITION_RATIO = 0.95
MIN_ORDER_VALUE = 2000

# z-thresholds (easy to trigger)
OPEN_Z = -0.5     # low vol => long
CLOSE_Z = 0.5     # high vol => flat

def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark(BENCHMARK)

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type="stock")
    set_slippage(FixedSlippage(0.002))

    g.last_state = "FLAT"

    # exactly like SVR: rebalance periodically so trades appear
    run_monthly(rebalance, 1, time="9:40", reference_security=BENCHMARK)
    run_daily(record_state, time="after_close")

def _close_series(count, end_date=None):
    df = get_price(UNDERLYING, end_date=end_date, count=count, frequency="1d", fields=["close"])
    if df is None or len(df) < count:
        return None
    return df["close"]

def _hv_and_z(decision_date):
    close = _close_series(HV_WINDOW + Z_WINDOW + 1, end_date=decision_date)
    if close is None:
        return np.nan, np.nan

    r = np.log(close / close.shift(1)).dropna()
    if len(r) < HV_WINDOW + Z_WINDOW:
        return np.nan, np.nan

    hv_series = r.rolling(HV_WINDOW).std() * math.sqrt(252)
    hv = float(hv_series.iloc[-1])

    mu = float(hv_series.iloc[-Z_WINDOW:].mean())
    sd = float(hv_series.iloc[-Z_WINDOW:].std())
    z = 0.0 if sd <= 1e-12 else (hv - mu) / sd
    return hv, z

def rebalance(context):
    today = context.current_dt.date()
    decision_date = context.previous_date  # avoid future

    hv, z = _hv_and_z(decision_date)
    if np.isnan(hv):
        log.info("Rebalance %s: no hv, skip" % str(today))
        return

    # signal (guaranteed to flip sometimes)
    state = g.last_state
    if z <= OPEN_Z:
        state = "LONG"
    elif z >= CLOSE_Z:
        state = "FLAT"
    else:
        # keep last state in middle zone
        state = g.last_state

    total_value = context.portfolio.total_value
    invest_value = total_value * POSITION_RATIO

    pos = context.portfolio.positions.get(UNDERLYING, None)
    cur_value = 0.0 if pos is None else pos.value

    if state == "LONG":
        target_value = invest_value
    else:
        target_value = 0.0

    # avoid tiny orders
    if abs(target_value - cur_value) < MIN_ORDER_VALUE:
        g.last_state = state
        log.info("Rebalance %s HV=%.3f Z=%.2f State=%s (no trade small diff)" % (str(today), hv, z, state))
        return

    order_target_value(UNDERLYING, target_value)
    g.last_state = state
    log.info("Rebalance %s HV=%.3f Z=%.2f State=%s Target=%.0f" % (str(today), hv, z, state, target_value))

def record_state(context):
    record(
        portfolio_value=context.portfolio.total_value,
        hold=int(UNDERLYING in context.portfolio.positions),
        cash=context.portfolio.cash
    )

