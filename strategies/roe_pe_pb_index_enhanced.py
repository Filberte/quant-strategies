# -*- coding: utf-8 -*-
# 回测记录（聚宽）：2010-01-01 ~ 2020-01-01，策略收益 27.50%，基准收益 17.44%，最大回撤 30.18%
from jqdata import *
import pandas as pd
import numpy as np
from datetime import timedelta

# ========= Params =========
INDEX_CODE = '000905.XSHG'   # CSI 500
MAX_HOLD = 10               # max holdings <= 10
UNIVERSE_N = 50             # top N by market cap
# =========================


def initialize(context):
    set_benchmark(INDEX_CODE)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')

    # rebalance monthly
    run_monthly(rebalance, 1, time='open')
    # record daily for plotting
    run_daily(daily_record, time='after_close')

    g.last_target = []
    g.rebalance_flag = 0
    g.model_ready = 0
    g.skip_count = 0


def rebalance(context):
    today = context.current_dt.date()
    g.rebalance_flag = 1
    log.info('===== rebalance {} ====='.format(today))
    log.info('portfolio value: {:.2f}'.format(context.portfolio.total_value))

    # 1) universe: CSI500 constituents
    try:
        universe = get_index_stocks(INDEX_CODE, date=today)
    except Exception as e:
        log.info('Error getting index stocks: {}'.format(e))
        universe = []

    log.info('original universe: {}'.format(len(universe)))

    if len(universe) < 10:
        log.info('universe too small, skip')
        g.skip_count += 1
        return

    # 2) 直接获取基本面数据，不过滤停牌等
    df_now = fetch_fundamentals_simple(universe, today)
    if df_now is None or df_now.empty:
        log.info('no fundamentals, skip')
        g.skip_count += 1
        return

    log.info('fundamentals shape: {}'.format(df_now.shape))

    # 选择市值最大的UNIVERSE_N只股票
    df_now = df_now.sort_values('market_cap', ascending=False).head(UNIVERSE_N)
    stocks_now = df_now.index.tolist()
    log.info('selected {} stocks by market cap'.format(len(stocks_now)))

    # 3) 简单的基本面因子排名
    try:
        # 确保有必要的列
        if 'roe' not in df_now.columns:
            df_now['roe'] = 0.0
        if 'pe_ratio' not in df_now.columns:
            df_now['pe_ratio'] = 100.0
        if 'pb_ratio' not in df_now.columns:
            df_now['pb_ratio'] = 5.0

        # 处理极端值
        df_now['pe_ratio'] = np.clip(df_now['pe_ratio'], 1, 200)
        df_now['pb_ratio'] = np.clip(df_now['pb_ratio'], 0.5, 50)

        # 计算估值倒数
        df_now['inv_pe'] = 1.0 / df_now['pe_ratio']
        df_now['inv_pb'] = 1.0 / df_now['pb_ratio']

        # 对ROE和估值因子进行标准化
        df_now['roe_norm'] = (df_now['roe'] - df_now['roe'].mean()) / df_now['roe'].std()
        df_now['inv_pe_norm'] = (df_now['inv_pe'] - df_now['inv_pe'].mean()) / df_now['inv_pe'].std()
        df_now['inv_pb_norm'] = (df_now['inv_pb'] - df_now['inv_pb'].mean()) / df_now['inv_pb'].std()

        # 避免无穷值
        df_now['roe_norm'] = df_now['roe_norm'].replace([np.inf, -np.inf], 0).fillna(0)
        df_now['inv_pe_norm'] = df_now['inv_pe_norm'].replace([np.inf, -np.inf], 0).fillna(0)
        df_now['inv_pb_norm'] = df_now['inv_pb_norm'].replace([np.inf, -np.inf], 0).fillna(0)

        # 综合评分
        df_now['score'] = (
            0.5 * df_now['roe_norm'] +
            0.25 * df_now['inv_pe_norm'] +
            0.25 * df_now['inv_pb_norm']
        )

        # 按得分排序
        df_now = df_now.sort_values('score', ascending=False)

        # 选择前MAX_HOLD只股票
        target = df_now.head(MAX_HOLD).index.tolist()
        g.last_target = target

        log.info('target holdings ({}): {}'.format(len(target), target))
        if len(df_now) > 0:
            log.info('top 5 scores:')
            for i, (idx, row) in enumerate(df_now.head(5).iterrows()):
                log.info('  {}. {}: {:.4f} (roe: {:.2f}, PE: {:.2f})'.format(
                    i+1, idx, row['score'], row['roe'], row['pe_ratio']))

        g.model_ready = 1

    except Exception as e:
        log.info('Error in scoring: {}'.format(e))
        # 如果评分失败，直接选择市值最大的股票
        target = df_now.head(MAX_HOLD).index.tolist()
        g.last_target = target
        log.info('Using simple market cap selection: {}'.format(target))

    # 4) 卖出非目标持仓
    current_positions = list(context.portfolio.positions.keys())
    log.info('current positions ({}): {}'.format(len(current_positions), current_positions))

    sell_list = [s for s in current_positions if s not in target]
    if sell_list:
        log.info('selling {} positions: {}'.format(len(sell_list), sell_list))
        for s in sell_list:
            order_target(s, 0)

    # 5) 买入目标持仓
    if target:
        total_value = context.portfolio.total_value
        cash = context.portfolio.cash

        # 使用95%的资金
        investable_cash = cash * 0.95
        if investable_cash < 1000:  # 最少需要1000元
            log.info('insufficient cash for investment: {:.2f}'.format(investable_cash))
            return

        each_value = investable_cash / len(target)
        log.info('each target value: {:.2f}'.format(each_value))

        orders_placed = 0
        for s in target:
            try:
                # 检查股票是否可以交易
                current_price = get_price(s, end_date=today, count=1, fields=['close'])
                if current_price is None or current_price.empty or current_price.iloc[0]['close'] <= 0:
                    log.info('{} has invalid price, skipping'.format(s))
                    continue

                # 计算目标股数
                price = current_price.iloc[0]['close']
                target_amount = int(each_value / price / 100) * 100  # 整手

                if target_amount <= 0:
                    continue

                # 获取当前持仓
                current_amount = 0
                if s in context.portfolio.positions:
                    current_amount = context.portfolio.positions[s].total_amount

                # 计算需要买入的股数
                amount_to_buy = target_amount - current_amount

                if amount_to_buy > 0:
                    order_result = order(s, amount_to_buy)
                    if order_result:
                        orders_placed += 1
                        log.info('bought {} shares of {}'.format(amount_to_buy, s))
                    else:
                        log.info('failed to buy {} shares of {}'.format(amount_to_buy, s))
                elif amount_to_buy < 0:
                    order_result = order(s, amount_to_buy)  # 负数表示卖出
                    if order_result:
                        log.info('sold {} shares of {}'.format(-amount_to_buy, s))

            except Exception as e:
                log.info('Error ordering {}: {}'.format(s, e))
                continue

        log.info('placed {} buy orders'.format(orders_placed))

    else:
        log.info('empty target list, no buy orders')

    log.info('rebalance completed')


def daily_record(context):
    """记录每日信息"""
    record(
        portfolio_value=context.portfolio.total_value,
        hold_num=len(context.portfolio.positions),
        rebalance_flag=g.rebalance_flag,
        cash=context.portfolio.cash,
        model_ready=g.model_ready,
        skip_count=g.skip_count
    )
    g.rebalance_flag = 0


def fetch_fundamentals_simple(stocks, date_):
    """获取基本面数据 - 简化版本"""
    if not stocks:
        return None

    try:
        # 简化查询，只获取必要字段
        q = query(
            valuation.code,
            valuation.market_cap,
            valuation.pe_ratio,
            valuation.pb_ratio,
            indicator.roe
        ).filter(
            valuation.code.in_(stocks)
        )

        df = get_fundamentals(q, date=date_)
        if df is None or df.empty:
            log.info('No fundamentals data for {}'.format(date_))
            return None

        df = df.set_index('code')

        # 重命名列
        df = df.rename(columns={
            'pe_ratio': 'pe_ratio',
            'pb_ratio': 'pb_ratio'
        })

        # 清理数据
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 填充NaN
        df = df.fillna(df.median())
        df = df.fillna(0)

        log.info('Fundamentals fetched: {} rows'.format(len(df)))
        return df

    except Exception as e:
        log.info('Error fetching fundamentals: {}'.format(e))
        return None


# 添加一个简单的价格检查函数
def is_tradable(stock, date):
    """检查股票是否可以交易"""
    try:
        # 获取最近几天的价格
        prices = get_price(stock, end_date=date, count=5, fields=['close', 'volume'])
        if prices is None or prices.empty:
            return False

        # 检查是否有价格和成交量
        if prices['close'].iloc[-1] <= 0 or prices['volume'].iloc[-1] <= 0:
            return False

        return True
    except:
        return False


# 添加一个超简单的策略用于测试
def test_rebalance(context):
    """测试用策略，直接买入指数成分股"""
    today = context.current_dt.date()
    log.info('===== test rebalance {} ====='.format(today))

    # 获取指数成分股
    try:
        stocks = get_index_stocks(INDEX_CODE, date=today)
    except:
        stocks = []

    if len(stocks) < 5:
        return

    # 只取前5只股票
    target = stocks[:5]

    # 卖出不在目标中的持仓
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target:
            order_target(stock, 0)

    # 买入目标股票
    if target:
        cash_per_stock = context.portfolio.cash * 0.8 / len(target)

        for stock in target:
            try:
                # 获取当前价格
                price_data = get_price(stock, end_date=today, count=1, fields=['close'])
                if price_data is None:
                    continue

                price = price_data.iloc[0]['close']
                if price <= 0:
                    continue

                # 计算购买数量
                amount = int(cash_per_stock / price / 100) * 100
                if amount > 0:
                    order(stock, amount)
                    log.info('Test order: {} shares of {}'.format(amount, stock))
            except:
                continue

    log.info('Test rebalance completed')

