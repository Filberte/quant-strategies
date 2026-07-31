# -*- coding: utf-8 -*-
"""
波动率曲面凸度套利策略 - JoinQuant完整解决方案
修复版：使用正确的证券代码
"""

from jqdata import *
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
# 使用有效的JoinQuant证券代码
g.target_future = 'IF88'  # 沪深300股指期货连续合约
# 或者使用具体合约，例如：'IF2309.CCFX'（IF2309合约）

g.vol_surface = None            # 波动率曲面对象
g.arbitrage_positions = {}      # 当前套利持仓
g.last_update_date = None       # 上次更新曲面日期
g.convexity_threshold = 0.3     # 凸度异常阈值
g.max_position_ratio = 0.15     # 单品种最大仓位比例
g.risk_free_rate = 0.03         # 无风险利率
g.option_data_cache = {}        # 期权数据缓存

# ==================== 辅助函数：获取有效期货合约 ====================
def get_valid_future_contract(future_code='IF', context=None):
    """
    获取当前有效的期货合约
    future_code: 期货品种代码，如 'IF'（沪深300）、'IC'（中证500）、'IH'（上证50）
    """
    try:
        # 方法1：获取主力合约
        if future_code == 'IF':
            # 沪深300股指期货
            return '510300.XSHG'  # 连续合约

        # 方法2：动态获取当前主力合约
        if context:
            current_date = context.current_dt.strftime('%Y-%m-%d')
            # 获取所有IF合约
            all_contracts = get_all_securities(types=['futures'], date=current_date)
            if_contracts = all_contracts[all_contracts['display_name'].str.contains('IF')]

            if not if_contracts.empty:
                # 按成交量排序，选择主力合约
                return if_contracts.index[0]

        # 方法3：使用具体合约（需要定期更换）
        return 'IF2309.CCFX'  # 示例：IF2309合约

    except Exception as e:
        log.error(f"获取期货合约失败: {e}")
        return '000300.XSHG'  # 回退到沪深300指数

# ==================== 1. 波动率曲面核心类 ====================
class VolatilitySurface3D:
    """三维波动率曲面建模"""

    def __init__(self, S0, current_date):
        """
        初始化曲面
        S0: 标的当前价格
        current_date: 当前日期
        """
        self.S0 = S0
        self.current_date = current_date
        self.maturities = []      # 期限（天数）
        self.moneyness = []       # 虚实程度（K/S）
        self.vol_matrix = None    # 波动率矩阵
        self.interpolator = None  # 插值函数

    def build_surface(self, option_data):
        """
        构建波动率曲面
        option_data: DataFrame，包含期权信息
        """
        if option_data.empty:
            return False

        # 提取唯一期限和虚实程度
        maturities_set = sorted(set(option_data['days_to_maturity']))
        moneyness_set = sorted(set(option_data['moneyness']))

        # 创建波动率矩阵
        vol_matrix = np.full((len(maturities_set), len(moneyness_set)), np.nan)

        for idx, row in option_data.iterrows():
            i = maturities_set.index(row['days_to_maturity'])
            j = moneyness_set.index(row['moneyness'])
            vol_matrix[i, j] = row['implied_vol']

        # 填充缺失值（使用最近邻插值）
        self._fill_missing_values(vol_matrix)

        self.maturities = np.array(maturities_set)
        self.moneyness = np.array(moneyness_set)
        self.vol_matrix = vol_matrix

        # 创建插值器
        self.interpolator = RegularGridInterpolator(
            (self.maturities, self.moneyness),
            self.vol_matrix,
            method='cubic',
            bounds_error=False,
            fill_value=None
        )

        return True

    def _fill_missing_values(self, matrix):
        """填充缺失值"""
        mask = np.isnan(matrix)
        if not mask.any():
            return

        # 使用最近的有效值填充
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                if np.isnan(matrix[i, j]):
                    # 查找最近的有效值
                    distances = []
                    values = []
                    for x in range(rows):
                        for y in range(cols):
                            if not np.isnan(matrix[x, y]):
                                dist = np.sqrt((x-i)**2 + (y-j)**2)
                                distances.append(dist)
                                values.append(matrix[x, y])

                    if values:
                        # 使用距离加权平均
                        weights = 1.0 / (np.array(distances) + 1e-6)
                        matrix[i, j] = np.average(values, weights=weights)

    def get_volatility(self, T, K):
        """
        获取指定期限和行权价的波动率
        T: 剩余天数
        K: 行权价
        """
        if self.interpolator is None:
            return None

        moneyness_val = K / self.S0
        point = np.array([[T, moneyness_val]])

        try:
            vol = float(self.interpolator(point)[0])
            return max(0.01, min(0.8, vol))  # 限制在合理范围
        except:
            return None

    def calculate_convexity(self, T, K, delta=0.02):
        """
        计算波动率凸度（对行权价的二阶导数）
        """
        vol_center = self.get_volatility(T, K)
        if vol_center is None:
            return 0

        # 有限差分法计算二阶导数
        vol_up = self.get_volatility(T, K * (1 + delta))
        vol_down = self.get_volatility(T, K * (1 - delta))

        if vol_up is None or vol_down is None:
            return 0

        convexity = (vol_up - 2 * vol_center + vol_down) / (delta**2)
        return convexity

    def calculate_smile_curvature(self, T):
        """
        计算波动率微笑曲率
        """
        moneyness_points = np.linspace(0.8, 1.2, 9)
        vols = []

        for m in moneyness_points:
            vol = self.get_volatility(T, self.S0 * m)
            if vol is not None:
                vols.append(vol)

        if len(vols) < 3:
            return 0

        # 二次拟合计算曲率
        coeffs = np.polyfit(moneyness_points[:len(vols)], vols, 2)
        return abs(coeffs[0]) * 100  # 放大100倍便于观察

    def plot_surface_info(self):
        """输出曲面信息"""
        if self.vol_matrix is None:
            print("曲面未构建")
            return

        log.info(f"标的价格: {self.S0:.2f}")
        log.info(f"期限范围: {self.maturities.min()} - {self.maturities.max()} 天")
        log.info(f"虚实程度: {self.moneyness.min():.2f} - {self.moneyness.max():.2f}")
        log.info(f"波动率范围: {self.vol_matrix.min():.3f} - {self.vol_matrix.max():.3f}")

        # 计算关键点凸度
        key_points = [(30, 0.95), (60, 1.0), (90, 1.05)]
        for T, m in key_points:
            convexity = self.calculate_convexity(T, self.S0 * m)
            log.info(f"T={T}d, K/S={m}: 凸度={convexity:.4f}")

# ==================== 2. 期权定价工具类 ====================
class OptionPricer:
    """期权定价工具"""

    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
        Black-Scholes期权定价
        T: 年化时间
        """
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0)

    @staticmethod
    def calculate_iv(price, S, K, T, r, option_type='call'):
        """计算隐含波动率"""
        if price <= 0:
            return 0.2  # 默认值

        def iv_objective(sigma):
            return OptionPricer.black_scholes(S, K, T, r, sigma, option_type) - price

        try:
            iv = brentq(iv_objective, 0.001, 2.0)
            return iv
        except:
            return 0.2

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        """计算希腊字母"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }

# ==================== 3. 套利检测引擎 ====================
class ConvexityArbitrageDetector:
    """凸度套利检测器"""

    def __init__(self, vol_surface, threshold=0.3):
        self.vol_surface = vol_surface
        self.threshold = threshold
        self.pricer = OptionPricer()

    def detect_arbitrage(self):
        """检测套利机会"""
        if self.vol_surface is None or self.vol_surface.interpolator is None:
            return []

        S0 = self.vol_surface.S0
        opportunities = []

        # 扫描关键期限
        maturities_to_check = [30, 60, 90, 180]
        moneyness_to_check = [0.9, 0.95, 1.0, 1.05, 1.1]

        for T in maturities_to_check:
            for m in moneyness_to_check:
                K = S0 * m
                convexity = self.vol_surface.calculate_convexity(T, K)

                # 检测异常凸度
                if abs(convexity) > self.threshold:
                    vol = self.vol_surface.get_volatility(T, K)

                    # 计算理论价格和套利空间
                    theoretical_price = self._calculate_theoretical_value(T, K, vol)

                    opportunity = {
                        'maturity_days': T,
                        'strike': K,
                        'moneyness': m,
                        'convexity': convexity,
                        'implied_vol': vol,
                        'theoretical_price': theoretical_price,
                        'signal_type': 'sell' if convexity > 0 else 'buy',
                        'confidence': min(abs(convexity) / 0.5, 1.0)  # 置信度0-1
                    }

                    opportunities.append(opportunity)

        # 按置信度排序
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:5]  # 返回前5个最佳机会

    def _calculate_theoretical_value(self, T, K, vol):
        """计算理论套利价值"""
        S0 = self.vol_surface.S0
        T_year = T / 365

        # 使用波动率曲面计算公平波动率
        fair_vol = vol

        # 考虑凸度调整
        convexity_adj = self.vol_surface.calculate_convexity(T, K) * 0.01

        # 调整后波动率
        adj_vol = fair_vol * (1 - convexity_adj)
        adj_vol = max(0.05, min(0.8, adj_vol))

        return adj_vol

# ==================== 4. 数据获取模块（修复版） ====================
def fetch_market_data(security, context):
    """获取市场数据 - 修复版"""
    try:
        # 获取标的期货数据
        # 先检查证券代码是否有效
        current_data = get_current_data()

        if security not in current_data:
            log.warn(f"证券代码 {security} 无效，尝试获取替代标的")
            # 使用沪深300指数作为替代
            security = '000300.XSHG'

        # 获取历史价格数据
        futures_data = get_price(security,
                                count=100,
                               end_date=context.current_dt,
                               frequency='daily',
                               fields=['close', 'volume'])

        # 计算历史波动率
        if len(futures_data) > 20:
            returns = futures_data['close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)
        else:
            hist_vol = 0.2

        current_price = futures_data['close'].iloc[-1] if len(futures_data) > 0 else 100

        return {
            'current_price': current_price,
            'historical_vol': hist_vol,
            'volume': futures_data['volume'].iloc[-1] if len(futures_data) > 0 else 0
        }

    except Exception as e:
        log.error(f"获取市场数据失败: {e}")
        # 返回默认值
        return {
            'current_price': 100,
            'historical_vol': 0.2,
            'volume': 0
        }

def generate_option_data(S0, current_date):
    """生成模拟期权数据（实际交易中需要从API获取）"""
    # 模拟不同期限和行权价的期权数据
    option_records = []

    maturities = [30, 60, 90, 180]
    moneyness_levels = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]

    # 基础波动率曲面参数
    base_vol = 0.2
    smile_strength = 0.1
    term_structure = 0.05

    for T in maturities:
        T_year = T / 365

        for m in moneyness_levels:
            K = S0 * m

            # 模拟波动率微笑
            moneyness_factor = (m - 1.0) ** 2
            vol_adjust = smile_strength * moneyness_factor

            # 模拟期限结构
            term_adjust = term_structure * (1 - np.exp(-T_year))

            # 总波动率
            total_vol = base_vol + vol_adjust + term_adjust
            total_vol = max(0.1, min(0.5, total_vol))

            # 添加随机噪声
            random_noise = np.random.normal(0, 0.02)
            total_vol += random_noise

            # 计算期权价格
            pricer = OptionPricer()
            call_price = pricer.black_scholes(S0, K, T_year, g.risk_free_rate, total_vol, 'call')
            put_price = pricer.black_scholes(S0, K, T_year, g.risk_free_rate, total_vol, 'put')

            option_records.append({
                'days_to_maturity': T,
                'moneyness': m,
                'strike': K,
                'call_price': call_price,
                'put_price': put_price,
                'implied_vol': total_vol,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(1000, 5000)
            })

    return pd.DataFrame(option_records)

# ==================== 5. 简化版交易管理器 ====================
class SimpleTradeManager:
    """简化版交易管理"""

    def __init__(self, context):
        self.context = context

    def execute_signal(self, security, signal_type, confidence):
        """执行交易信号"""
        try:
            current_price = get_current_data()[security].last_price

            # 计算交易金额
            total_value = self.context.portfolio.total_value
            trade_value = total_value * 0.1 * confidence  # 10%基础仓位 × 置信度

            if trade_value < 10000:  # 小于1万不交易
                return None

            if signal_type == 'buy':
                order_value(security, trade_value)
                log.info(f"买入 {security}: 金额={trade_value:.2f}, 价格={current_price:.2f}")
                return {'action': 'buy', 'value': trade_value, 'price': current_price}
            else:
                # 检查是否有持仓可卖
                if security in self.context.portfolio.positions:
                    position = self.context.portfolio.positions[security]
                    if position.closeable_amount > 0:
                        order_target_value(security, position.value - trade_value)
                        log.info(f"卖出 {security}: 金额={trade_value:.2f}")
                        return {'action': 'sell', 'value': trade_value, 'price': current_price}

        except Exception as e:
            log.error(f"交易执行失败: {e}")

        return None

# ==================== 6. 主要策略函数（修复版） ====================
def initialize(context):
    """初始化策略"""
    # 设置基准
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    # 设置手续费和滑点
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        close_today_commission=0,
        min_commission=5
    ), type='stock')

    set_slippage(PriceRelatedSlippage(0.002))

    # 初始化全局变量
    g.vol_surface = None
    g.arbitrage_positions = {}
    g.trade_manager = SimpleTradeManager(context)

    # 获取有效的期货合约
    g.target_future = get_valid_future_contract('IF', context)
    log.info(f"使用标的: {g.target_future}")

    # 定时运行
    run_daily(before_market_open, time='9:00')
    run_daily(run_strategy, time='14:30')
    run_daily(after_market_close, time='15:30')

    log.info('=== 波动率曲面凸度套利策略初始化完成 ===')
    log.info(f'目标标的: {g.target_future}')
    log.info(f'凸度阈值: {g.convexity_threshold}')
    log.info(f'最大仓位比例: {g.max_position_ratio}')

def before_market_open(context):
    """开盘前准备"""
    log.info(f'交易日: {context.current_dt.date()}')

    # 获取市场数据
    market_data = fetch_market_data(g.target_future, context)
    log.info(f'标的价格: {market_data["current_price"]:.2f}, 历史波动率: {market_data["historical_vol"]:.3f}')

def run_strategy(context):
    """主策略逻辑"""
    current_date = context.current_dt.date()

    # 每周更新一次波动率曲面
    if g.last_update_date is None or (current_date - g.last_update_date).days >= 7:
        log.info("更新波动率曲面...")
        update_volatility_surface(context)
        g.last_update_date = current_date

    # 检测套利机会
    if g.vol_surface is not None:
        detector = ConvexityArbitrageDetector(g.vol_surface, g.convexity_threshold)
        opportunities = detector.detect_arbitrage()

        if opportunities:
            log.info(f"发现 {len(opportunities)} 个套利机会")

            # 执行最佳机会
            if opportunities:
                best_opp = opportunities[0]

                # 使用期货作为替代标的
                target_security = g.target_future

                # 根据信号类型执行
                if best_opp['signal_type'] == 'buy':
                    g.trade_manager.execute_signal(target_security, 'buy', best_opp['confidence'])
                else:
                    g.trade_manager.execute_signal(target_security, 'sell', best_opp['confidence'])

                log.info(f"执行{best_opp['signal_type']}信号: T={best_opp['maturity_days']}d, "
                        f"凸度={best_opp['convexity']:.4f}, 置信度={best_opp['confidence']:.2f}")
        else:
            log.info("未发现明显套利机会")

    # 记录投资组合状态
    log_portfolio_status(context)

def update_volatility_surface(context):
    """更新波动率曲面"""
    # 获取当前标的价格
    market_data = fetch_market_data(g.target_future, context)
    S0 = market_data['current_price']

    if S0 <= 0:
        log.error("获取标的价格失败")
        return

    # 生成/获取期权数据
    option_data = generate_option_data(S0, context.current_dt)

    # 构建波动率曲面
    g.vol_surface = VolatilitySurface3D(S0, context.current_dt)
    success = g.vol_surface.build_surface(option_data)

    if success:
        log.info("波动率曲面构建成功")
        g.vol_surface.plot_surface_info()
    else:
        log.error("波动率曲面构建失败")

def after_market_close(context):
    """收盘后处理"""
    log.info(f"交易日结束: {context.current_dt.date()}")

    # 记录当日盈亏
    total_pnl = context.portfolio.daily_pnl
    log.info(f"当日盈亏: {total_pnl:.2f}")

    # 记录持仓信息
    if context.portfolio.positions:
        for security, position in context.portfolio.positions.items():
            pnl = position.value - position.cost_basis
            log.info(f"{security}: 数量={position.total_amount}, "
                    f"价值={position.value:.2f}, 盈亏={pnl:.2f}")

    log.info('=' * 50)

def log_portfolio_status(context):
    """记录投资组合状态"""
    portfolio = context.portfolio

    log.info(f"总资产: {portfolio.total_value:.2f}")
    log.info(f"可用资金: {portfolio.available_cash:.2f}")
    log.info(f"持仓数量: {len(portfolio.positions)}")

    # 计算风险指标
    if portfolio.starting_cash > 0:
        total_return = (portfolio.total_value - portfolio.starting_cash) / portfolio.starting_cash
        log.info(f"累计收益率: {total_return:.2%}")

# ==================== 7. 备用策略版本（使用股票代替期货） ====================
def initialize_stock_version(context):
    """股票版本策略（如果期货不可用）"""
    # 设置基准
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    # 设置手续费
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0.001,
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5
    ), type='stock')

    # 使用沪深300ETF代替期货
    g.target_security = '510300.XSHG'  # 华泰柏瑞沪深300ETF

    # 其他初始化代码...
    run_daily(run_strategy, time='14:30')

# ==================== 有效的JoinQuant证券代码示例 ====================
"""
股票：
- '000001.XSHE'  # 平安银行
- '510300.XSHG'  # 沪深300ETF

指数：
- '000300.XSHG'  # 沪深300指数

期货连续合约：
- 'IF88'         # 沪深300主力连续
- 'IC88'         # 中证500主力连续
- 'IH88'         # 上证50主力连续

具体期货合约：
- 'IF2309.CCFX'  # IF2309合约
- 'IC2309.CCFX'  # IC2309合约

注意：在JoinQuant中，获取期货数据需要相应的权限
"""

