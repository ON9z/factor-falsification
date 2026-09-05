"""因子库 — LLM 研究员可引用的「原子因子」注册表。

每个原子因子是一块预计算积木(LLM 只能引用/组合这些积木, 不能自由写 Python, 安全+快)。
每个因子带三要素:
  - description:     喂给 LLM 的数据字典(它据此提假设)
  - available_since: 该因子底层数据的覆盖起始日(YYYY-MM-DD), 回测器只在覆盖区间内算,
                     否则早期段 turnover=0/NaN 会被当成真信号(今日已踩过的坑)。
  - compute(df, **params) -> Series: 计算函数, 输入长格式 qfq 日线(按 code 分组、date 升序),
                     返回与 df 同 index 的因子值。

无泄露铁律(对应 [[finagent-silent-bug-sweep-20260828]] 的 ML 泄漏):
  - 时间序列统计一律用 .rolling/.shift 只取过去(因果), 严禁全样本 mean/std 回填。
  - 横截面统计一律在同一天内(rank/z-score 只用自己的横截面), 不跨日。
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── 数据覆盖起始日(来自 08-29 数据侦察) ──────────────────────────────────────
AVAILABLE_SINCE = {
    # 09-02 回补 10 年(用户指令「所有数据回补10年才为稳妥」): 以下域已扩到 2016。
    # 未扩域: valuation_baostock(ps/pcf 市销率市现率, baostock 源仅1年, value 已证伪) /
    #         moneyflow(资金流已证伪+tushare口径需重映射) / limitup(封板时间仅1年)。
    "price":   "2016-01-04",   # stock_daily_qfq 前复权日线(价格/量/换手), 10 年(09-02 回补 2016, min=2016-01-04)
    "valuation": "2016-01-04", # stock_daily_qfq.pe_ttm/pb(tushare daily_basic, 10 年)
    "size": "2016-01-04",      # stock_daily_qfq.total_mv/circ_mv(tushare daily_basic, 10 年)
    "valuation_baostock": "2025-08-01", # valuation_baostock_hist(ps_ttm/pcf_ncf 仍只 1 年, value 已证伪)
    "moneyflow": "2024-01-02", # money_flow_daily, 已回补 2.6 年(新浪 zjlrqs); 资金流已证伪, 未扩 10 年
    "dragontiger": "2016-01-04",  # dragon_tiger_seat, 10 年(09-02 回补 2016)
    "limitup": "2025-08-14",   # limitup_seal(封板时间), 1 年
    "insider": "2016-01-04",   # insider_trades 股东增减持(tushare, 10 年, min=2016-01-01)
    "fundamental": "2016-04-30",  # fundamental_indicators(2016-03-31 首份报告, _report_avail 披露截止→04-30 才可用)
    "analyst": "2016-01-04",   # analyst_rating 研报评级(巨潮, 10 年, min=2016-01-01; 目标价仍仅近期研报有)
}


# ── 计算辅助 ────────────────────────────────────────────────────────────────

def _ret(df: pd.DataFrame, days: int) -> pd.Series:
    """N 日价格收益率 close/close.shift(days)-1, 每股内部前向因果。"""
    g = df.groupby("code", sort=False)["close"]
    return g.shift(0) / g.shift(days) - 1.0


def _market_mean(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """把每股序列对齐到日期, 计算该日横截面等权均值(市场基准)。

    保留 df 的 index(而非 RangeIndex), 否则与 r(带 df index) 相减时错位。
    """
    tmp = pd.DataFrame({"date": df["date"].values, "v": s.values}, index=df.index)
    return tmp.groupby("date")["v"].transform("mean")


def _rel_ret(df: pd.DataFrame, days: int) -> pd.Series:
    """N 日相对强弱 = 个股收益 - 当日全市场等权均值收益。"""
    r = _ret(df, days)
    return r - _market_mean(r, df)


def _turnover_avg(df: pd.DataFrame, days: int) -> pd.Series:
    """N 日均换手率(活跃度)。"""
    return df.groupby("code", sort=False)["turnover"].transform(
        lambda s: s.rolling(days, min_periods=1).mean())


def _turnover_zscore(df: pd.DataFrame, days: int) -> pd.Series:
    """换手率相对自身 N 日历史的 z-score(量能异动), 因果(rolling 只用过去)。"""
    def _z(s: pd.Series) -> pd.Series:
        m = s.rolling(days, min_periods=days // 2).mean()
        sd = s.rolling(days, min_periods=days // 2).std()
        out = (s - m) / sd.replace(0, np.nan)
        return out
    return df.groupby("code", sort=False)["turnover"].transform(_z)


def _vol_ratio(df: pd.DataFrame, days: int) -> pd.Series:
    """N 日量比 = 近 N 日均量 / 近 60 日均量(放量)。"""
    v = df.groupby("code", sort=False)["volume"]
    short = v.transform(lambda s: s.rolling(days, min_periods=1).mean())
    long_ = v.transform(lambda s: s.rolling(60, min_periods=1).mean())
    return short / long_.replace(0, np.nan)


def _close_pos(df: pd.DataFrame, days: int) -> pd.Series:
    """收盘价在 N 日高低区间的相对位置(趋势强弱: 1=创新高, 0=创新低)。"""
    g = df.groupby("code", sort=False)
    hi = g["high"].transform(lambda s: s.rolling(days, min_periods=1).max())
    lo = g["low"].transform(lambda s: s.rolling(days, min_periods=1).min())
    rng = (hi - lo).replace(0, np.nan)
    return (df["close"] - lo) / rng


def _ret_vol(df: pd.DataFrame, days: int) -> pd.Series:
    """N 日风险调整收益 = N 日收益 / N 日收益标准差(动量质量)。"""
    r = _ret(df, days)
    sd = df.groupby("code", sort=False)["change_pct"].transform(
        lambda s: s.rolling(days, min_periods=days // 2).std()).replace(0, np.nan)
    return r / (sd / 100.0 + 1e-9)


def _gap(df: pd.DataFrame) -> pd.Series:
    """开盘跳空 gap = open/昨收(pre_close) - 1。负=低开, 正=高开。

    昨收用 stock_daily_qfq.pre_close(前复权昨收, 除权日口径正确; 勿用 close.shift(1)——
    后者跨复权基准在除权日会现假跳空, 已实测 12,438 行差异)。开盘价 09:25 集合竞价即知,
    当日开盘即可入场(回测器 entry_lag=0), 无 look-ahead。
    direction='low' 做多低开(负 gap 最大者) = 低开高走/低吸假设(海通/西部: 避开隔夜)。
    """
    pc = df["pre_close"].where(df["pre_close"] > 0)
    return df["open"] / pc - 1.0


def _positive_only(s: pd.Series) -> pd.Series:
    """估值比率只保留正值: 负 PE(亏损)/负 PB(资不抵债)/负 PCF(无现金流)不是「便宜」, 是「亏损」,
    若当低值=便宜会被误买亏损股, 故置 NaN 排除。"""
    return s.where(s > 0)


def _pe_ttm(df: pd.DataFrame) -> pd.Series:
    """滚动市盈率(股价/近12月每股收益)。低=便宜(价值)。负=亏损, 已排除。"""
    return _positive_only(df["pe_ttm"])


def _pb(df: pd.DataFrame) -> pd.Series:
    """市净率(股价/每股净资产)。低=便宜(价值)。负=资不抵债, 已排除。"""
    return _positive_only(df["pb"])


def _ps_ttm(df: pd.DataFrame) -> pd.Series:
    """市销率(股价/近12月每股营收)。低=便宜(价值)。"""
    return _positive_only(df["ps_ttm"])


def _pcf_ncf(df: pd.DataFrame) -> pd.Series:
    """市现率(股价/每股经营现金流)。低=便宜(价值)。负=无现金流, 已排除。"""
    return _positive_only(df["pcf_ncf"])


def _size(df: pd.DataFrame) -> pd.Series:
    """总市值 size 因子(万元, tushare daily_basic 口径)。低=小市值(小盘溢价), 高=大盘股。

    用 total_mv(总市值=学术 size 口径)而非 circ_mv(流通市值); 负/零(不应出现)置 NaN。
    横截面 rank 下 log 与否不变, 故直接返回原始市值, 方向 direction='low' = 做多小盘。
    """
    return df["total_mv"].where(df["total_mv"] > 0)


def _dt_inst_net(df: pd.DataFrame, days: int) -> pd.Series:
    """龙虎榜机构席位净买入强度 = 近 N 日机构专用席位净买入额(滞后1日) / 近 N 日成交额。

    look-ahead 铁律: 龙虎榜盘后(晚间)发布, 信号日 t 收盘时只见 t-1 的龙虎榜,
    故机构净买额整体 shift(1), 再 rolling 求和 → 分子只含 [t-N, t-1] 的机构净买,
    全部在 t 收盘前已知。分母用 amount(成交额, 收盘即知, 无需滞后)。
    机构净买额(元)/成交额(元) 量纲一致 → 主力净流入占比的龙虎榜版(资金承接/出货)。

    稀疏口径: 分子 min_periods=1 → 近 N 日无任何机构席位的股票得 NaN(不进横截面),
    只在「近 N 日上过龙虎榜且有机构席位」的股票子集内做横截面 rank。
    """
    g = df.groupby("code", sort=False)
    num = g["inst_net"].transform(lambda s: s.shift(1).rolling(days, min_periods=1).sum())
    den = g["amount"].transform(lambda s: s.rolling(days, min_periods=days // 2).sum())
    return num / den.replace(0, np.nan)


def _insider_net(df: pd.DataFrame, days: int) -> pd.Series:
    """股东增减持净强度 = 近 N 日净增持比例(增持-减持变动比例%之和)。

    look-ahead 铁律: 增减持公告日(ann_date)盘后出, 与龙虎榜 dt_inst_net 同 shift(1) 口径
    —— 信号日 t 收盘只用 [t-N, t-1] 的公告。insider_net 已在 _load_qfq 按 (code,date) 聚合为
    净 signed change_ratio(%): IN增持=+, DE减持=-。分子 min_periods=1 → 近 N 日无增减持披露的
    股票得 NaN(不进横截面), 只在「近期有增减持披露」子集内 rank(稀疏, 类似龙虎榜子集)。
    """
    g = df.groupby("code", sort=False)
    return g["insider_net"].transform(lambda s: s.shift(1).rolling(days, min_periods=1).sum())


def _main_net_flow(df: pd.DataFrame, days: int) -> pd.Series:
    """主力净流入强度 = 近 N 日主力净流入占比(%)的均值。

    主力=超大单+大单; main_net 已是「净流入额/成交额×100」的比例, 无需再除 amount。
    资金流向盘后即知(当日收盘已确定), 无 look-ahead, 无需 shift(区别于龙虎榜盘后发布)。
    高=近 N 日主力持续净流入(资金承接), 低=净流出(资金撤离)。稠密信号(全股票每日有值)。
    """
    return df.groupby("code", sort=False)["main_net"].transform(
        lambda s: s.rolling(days, min_periods=1).mean())


def _super_large_flow(df: pd.DataFrame, days: int) -> pd.Series:
    """超大单净流入强度 = 近 N 日超大单净流入占比(%)的均值。

    超大单=最粗粒度成交(偏机构/大资金), 比主力(含大单)信号更强但更稀疏。
    高=近 N 日超大单持续净流入。
    """
    return df.groupby("code", sort=False)["super_large"].transform(
        lambda s: s.rolling(days, min_periods=1).mean())


# ── 基本面因子(报告期, as-of join 已在 _load_qfq 做点-in-time 滞后) ───────────

def _roe(df: pd.DataFrame) -> pd.Series:
    """净资产收益率 ROE(最新已知财报)。高=盈利质量好(质量股)。"""
    return df["roe_avg"]


def _roa(df: pd.DataFrame) -> pd.Series:
    """总资产净利润率 ROA(最新已知财报)。高=资产运用效率高(质量)。"""
    return df["roa"]


def _net_margin(df: pd.DataFrame) -> pd.Series:
    """销售净利率%(最新已知财报)。高=盈利能力强(质量)。"""
    return df["net_margin"]


def _revenue_growth(df: pd.DataFrame) -> pd.Series:
    """主营业务收入增长率%同比(最新已知财报)。高=成长股。"""
    return df["revenue_growth"]


def _profit_growth(df: pd.DataFrame) -> pd.Series:
    """净利润增长率%同比(最新已知财报)。高=盈利成长。"""
    return df["profit_growth"]


def _debt_to_asset(df: pd.DataFrame) -> pd.Series:
    """资产负债率%(最新已知财报)。高=高杠杆(风险), 低=财务稳健(质量)。"""
    return df["debt_to_asset"]


# ── 研报评级/目标价因子(妙想 MCP 家族1, 稀疏机构事件) ─────────────────────

def _analyst_rating(df: pd.DataFrame) -> pd.Series:
    """研报评级共识分(1=买入..5=卖出, 越低越看多, 已前向填充至最新研报)。

    低=分析师最看多(analyst revision drift 稀疏事件, 同 dt_inst_net 龙虎榜家族)。
    无研报覆盖的股票为 NaN(不进横截面)。
    """
    return df["analyst_score"]


def _analyst_upside(df: pd.DataFrame) -> pd.Series:
    """研报目标价上行空间 = (目标最高价 − 现价)/现价(已前向填充)。高=分析师预期涨幅大。

    分母用 close_nom(未复权), 与目标价 max_price(未复权)同坐标系; 用 qfq close(前复权)会
    高估高股息/早期年份上行空间(复权口径错配, 09-04 修复, 见 [[finagent-analyst-upside-coverage]])。
    目标价覆盖 18-33%(巨潮源 2016-2026 均有)。
    """
    tp = df["analyst_tp"].where(df["analyst_tp"] > 0)
    return (tp - df["close_nom"]) / df["close_nom"]


def _rating_upgrade(df: pd.DataFrame, days: int) -> pd.Series:
    """近 N 日研报评级调高次数(analyst revision drift 稀疏机构事件, 妙想家族1)。

    高=近期被分析师上调评级最多(做多)。事件落在 signal_date(盘后发布), 因子近 N 日滚动求和;
    factor 的 forward 收益=次日开盘买, 天然对齐「盘后发布→次日可买」无 look-ahead。
    无调高事件=0(非缺失), 多数股票=0 → 做多高值组选中「近期有调高」的稀疏子集。
    """
    g = df.groupby("code", sort=False)
    return g["upgrade"].transform(lambda s: s.rolling(days, min_periods=1).sum())


# ── 广发金工扩展(09-03) ───────────────────────────────────────────────────

def _moneyflow_surge(df: pd.DataFrame, days: int) -> pd.Series:
    """主力净流入突变 = main_net 相对自身近 N 日历史的 z-score(因果 rolling 只用过去)。

    区别于 main_net_flow(近 N 日均值=水平, 已证伪=追高马甲 [[finagent-moneyflow-factor-dead]]):
    突变捕捉「今日净流入相对自身基线突然放大」的冲击分量 —— 广发金工口径是突变而非水平。
    正=今日主力净流入异常放大(疑似机构进场早期), 负=异常流出。数据源 money_flow_daily
    (2024-01 起 2.6 年, 10 年回补在跑), 盘后即知无 look-ahead。
    """
    def _z(s: pd.Series) -> pd.Series:
        m = s.rolling(days, min_periods=days // 2).mean()
        sd = s.rolling(days, min_periods=days // 2).std()
        return (s - m) / sd.replace(0, np.nan)
    return df.groupby("code", sort=False)["main_net"].transform(_z)


def _smart_analyst(df: pd.DataFrame, days: int) -> pd.Series:
    """聪明钱×分析师确认 = 龙虎榜机构净买入强度 × 分析师看多度。

    机构净买入 = _dt_inst_net(滞后1日, 稀疏子集); 分析师看多度 = 3 − analyst_score
    (1=买入..5=卖出, 前向填充), 映射到 [-2,+2](强买+2/中性0/卖出-2), 无研报覆盖=NaN(剔除)。
    乘积为正 = 机构真金白银买入 且 分析师背书(双确认); 为负 = 机构卖 或 分析师看空。
    覆盖 = dt_inst_net ∩ 分析师覆盖, 比单一 dt_inst_net 更窄 → 若 t 不升则交互无增量
    (广发「聪明钱×分析师」的诚实检验: 双确认是否放大最强单因子 dt_inst_net)。
    """
    inst = _dt_inst_net(df, days)
    bull = (3.0 - df["analyst_score"]).where(df["analyst_score"].notna())
    return inst * bull


# ── 因子注册表 ──────────────────────────────────────────────────────────────

FACTORS: dict[str, dict] = {
    "ret": {
        "description": "N日价格动量: close/close.shift(N)-1。正值=近期上涨(动量), 负值=近期下跌(反转)",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 1, "max": 60, "default": 5}},
        "compute": _ret,
    },
    "rel_ret": {
        "description": "N日相对强弱: 个股N日收益 - 当日全市场等权均值收益。正值=跑赢大盘(相对动量)",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 1, "max": 60, "default": 10}},
        "compute": _rel_ret,
    },
    "turnover_avg": {
        "description": "N日均换手率: 流动性/活跃度水平。高=交易活跃(易被资金关注), 低=冷门",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 1, "max": 60, "default": 5}},
        "compute": _turnover_avg,
    },
    "turnover_zscore": {
        "description": "换手率相对自身N日历史的z-score: 量能异动。>1=近期异常放量, <0=缩量",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 5, "max": 60, "default": 20}},
        "compute": _turnover_zscore,
    },
    "vol_ratio": {
        "description": "量比: 近N日均量/近60日均量。>1.5=显著放量(资金进场), <0.5=缩量",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 1, "max": 20, "default": 5}},
        "compute": _vol_ratio,
    },
    "close_pos": {
        "description": "收盘价在N日高低区间的相对位置(0~1): 1=创N日新高(强势), 0=创N日新低",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 5, "max": 120, "default": 20}},
        "compute": _close_pos,
    },
    "ret_vol": {
        "description": "N日风险调整收益: N日收益/N日波动率。高=高动量低波动(动量质量), 低=高波动弱动量",
        "available_since": AVAILABLE_SINCE["price"],
        "params": {"days": {"type": "int", "min": 5, "max": 60, "default": 20}},
        "compute": _ret_vol,
    },
    "gap": {
        "description": "开盘跳空 gap = open/昨收(pre_close) - 1。负=低开, 正=高开。开盘价集合竞价(09:25)即知, "
                       "当日开盘即可入场(回测器 entry_lag=0), 无 look-ahead。direction='low' 做多低开(负gap最大) "
                       "= 低开高走/低吸假设(海通/西部: 避开隔夜)。价格数据 10 年(09-02 回补 2016)",
        "available_since": "2016-01-04",
        "params": {},
        "compute": _gap,
    },
    "pe_ttm": {
        "description": "滚动市盈率(股价/近12月EPS)。低=便宜(价值股), 高=贵(成长股)。负值(亏损股)已排除。"
                       "源升级 08-31: qfq 表 tushare daily_basic 4 年(原 baostock 仅 1 年)。参数无关。"
                       "【回测08-31 FAIL】低PE无alpha(train ~0, test -0.34% t=-2.1): 价值=小市值马甲, 同pb",
        "available_since": AVAILABLE_SINCE["valuation"],
        "params": {},
        "compute": _pe_ttm,
    },
    "pb": {
        "description": "市净率(股价/每股净资产)。低=便宜(价值), 高=贵。负值(资不抵债)已排除。"
                       "源升级 08-31: qfq 表 4 年(原 baostock 仅 1 年)。参数无关。"
                       "【回测08-31 FAIL】4年窗口 train +0.25%(t=2.7)但 test -0.31%(t=-2.1)反转, sfc=0.5: "
                       "低PB价值=小市值马甲(与size相关), 独立无alpha, 印证 [[finagent-pandaai-5yr-value-only]] 价值P2归零",
        "available_since": AVAILABLE_SINCE["valuation"],
        "params": {},
        "compute": _pb,
    },
    "size": {
        "description": "总市值 size 因子(万元, tushare daily_basic 10 年)。低=小市值(小盘溢价), 高=大盘股。"
                       "direction='low' 做多小盘。"
                       "【09-02 10年重验】h=5/10/20 t=5.27/7.24/9.00, 但动态成本后净超额 h5/h10 转负(-0.61%/-0.21%), "
                       "仅 h=20 活(净+0.68% PASS)。诚实定性=β非α(小盘溢价经典因子溢价, 见 [[finagent-pandaai-size-dominates]]), "
                       "短窗口正超额是低估成本假象(见 [[finagent-honest-cost-downgrade]])",
        "available_since": AVAILABLE_SINCE["size"],
        "params": {},
        "compute": _size,
    },
    "ps_ttm": {
        "description": "市销率(股价/近12月每股营收)。低=便宜(价值)。⚠ 仍 baostock 源仅 1 年。参数无关",
        "available_since": AVAILABLE_SINCE["valuation_baostock"],
        "params": {},
        "compute": _ps_ttm,
    },
    "pcf_ncf": {
        "description": "市现率(股价/每股经营现金流)。低=便宜(价值)。负值(无现金流)已排除。⚠ 仍 baostock 源仅 1 年。参数无关",
        "available_since": AVAILABLE_SINCE["valuation_baostock"],
        "params": {},
        "compute": _pcf_ncf,
    },
    "dt_inst_net": {
        "description": "龙虎榜机构席位净买入强度: 近N日机构专用席位净买入额(滞后1日)/近N日成交额。"
                       ">0=机构净买入(资金承接), <0=机构净卖出(出货)。仅龙虎榜股有非零值, 其余为0(无机构席位信号)。"
                       "【09-02 10年重验】最强因子: h=5/10/20 t=8.45/9.33/7.46, 全 horizon 净超额正(+0.30%/+0.64%/+0.78%), "
                       "sfc=1.0 perm=0.01, 10 年比 2 年(t=4.7)更强。唯一 FAIL 闸门=覆盖 3.7%<20%(龙虎榜稀疏固有, "
                       "PASS 仅限「近 N 日上过龙虎榜且有机构席位」子集, 不外推)",
        "available_since": AVAILABLE_SINCE["dragontiger"],
        "params": {"days": {"type": "int", "min": 3, "max": 20, "default": 10}},
        "compute": _dt_inst_net,
    },
    "insider_net": {
        "description": "股东增减持净强度: 近N日净增持比例(增持-减持变动比例%之和, 公告日shift(1))。"
                       ">0=产业资本/高管净增持(真金白银看多), <0=净减持(看空/套现)。"
                       "稀疏信号(仅近期有增减持披露的股票有值), 区别于龙虎榜席位(游资/机构盘中席位)。"
                       "【回测08-31证伪】增持方向无alpha(增持太稀20%+多象征性, 扣费为负); "
                       "唯一活口=做多减持的弱反转(利空出尽反弹, 10日持有+0.4~0.7% t~3.5 但h=20归零=短窗口尾部), "
                       "非动量马甲(动量同窗已死)但幅度贴摩擦+多配置扫描风险, 达不到dt_inst_net级别, 暂不ship",
        "available_since": AVAILABLE_SINCE["insider"],
        "params": {"days": {"type": "int", "min": 3, "max": 60, "default": 20}},
        "compute": _insider_net,
    },
    "main_net_flow": {
        "description": "主力净流入强度: 近N日主力净流入占比(净流入/成交额×100)均值。"
                       "高=主力持续净流入(资金承接), 低=净流出(资金撤离)。稠密信号(全股票每日), 盘后即知无look-ahead",
        "available_since": AVAILABLE_SINCE["moneyflow"],
        "params": {"days": {"type": "int", "min": 1, "max": 30, "default": 5}},
        "compute": _main_net_flow,
    },
    "super_large_flow": {
        "description": "超大单净流入强度: 近N日超大单净流入占比(%)均值。超大单偏机构/大资金, 比主力信号更强",
        "available_since": AVAILABLE_SINCE["moneyflow"],
        "params": {"days": {"type": "int", "min": 1, "max": 30, "default": 5}},
        "compute": _super_large_flow,
    },
    "roe": {
        "description": "净资产收益率ROE(最新已知财报, 点-in-time滞后)。高=盈利质量好(质量股), 低=盈利差。"
                       "【回测08-31证伪】白马拥挤: high贴摩擦, low(劣质)显著跑输但A股不可做空, 无正alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _roe,
    },
    "roa": {
        "description": "总资产净利润率ROA(最新已知财报)。高=资产运用效率高(质量)。"
                       "【回测08-31证伪】同roe白马拥挤, 无正alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _roa,
    },
    "net_margin": {
        "description": "销售净利率%(最新已知财报)。高=盈利能力强(质量)。"
                       "【回测08-31证伪】同roe白马拥挤, 无正alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _net_margin,
    },
    "revenue_growth": {
        "description": "主营业务收入增长率%同比(最新已知财报)。高=成长股。"
                       "【09-02 10年证伪】10年窗口 t≈0(h=5/10/20 t=0.36/0.15/-0.01), sfc=0.5, 净超额-0.45%: "
                       "08-31 的 PASS(4年 t=4.7)是覆盖缺口假阳性, 非 regime 依赖而是纯噪声(见 "
                       "[[finagent-10yr-backfill-reverify]])。稠密季频因子无 alpha, 印证稀疏机构事件才有 alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _revenue_growth,
    },
    "profit_growth": {
        "description": "净利润增长率%同比(最新已知财报)。高=盈利成长。"
                       "【回测08-31证伪】利润增速无alpha(基数效应/一次性损益噪声), 仅营收增速(revenue_growth)有alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _profit_growth,
    },
    "debt_to_asset": {
        "description": "资产负债率%(最新已知财报)。高=高杠杆(风险), 低=财务稳健(质量, 做多低值)。"
                       "【回测08-31】高杠杆显著跑输(h=20 -0.82% t=-4.76)可做过滤器/排除项; 但做多低杠杆仅+0.1%无alpha",
        "available_since": AVAILABLE_SINCE["fundamental"],
        "params": {},
        "compute": _debt_to_asset,
    },
    "analyst_rating": {
        "description": "研报评级共识分(1=买入..5=卖出, 越低越看多, 前向填充至最新研报)。"
                       "做多低值=分析师最看多股(analyst revision drift 稀疏机构事件, 妙想MCP家族1)。"
                       "【09-02 10年重验见 reverify_10yr_factors.py】",
        "available_since": AVAILABLE_SINCE["analyst"],
        "params": {},
        "compute": _analyst_rating,
    },
    "analyst_upside": {
        "description": "研报目标价上行空间=(目标最高价-现价)/现价(前向填充)。高=分析师预期涨幅大。"
                       "【09-02 10年重验】慢因子: 仅 h=20 活(毛+0.88% 净+0.25% t=9.68 PASS), h=5/10 动态成本后净转负"
                       "(-0.46%/-0.23%, t=3.73/6.50)。4年 t=12.74 → 10年 h20 t=9.68 略降。"
                       "目标价覆盖全年 18-33%(巨潮源, 2016-2026 均有), 非仅近期; 但覆盖仍偏大盘(tilt, 见 "
                       "[[finagent-analyst-upside-coverage]])。"
                       "【09-04/05 复权修复】分母改 close_nom(未复权, 与目标价同坐标系); 修复后 h20 test +0.756%"
                       "(t=7.37) 存活, vs全市场真实(+0.74%/20日), 唯一活信号",
        "available_since": AVAILABLE_SINCE["analyst"],
        "params": {},
        "compute": _analyst_upside,
    },
    "rating_upgrade": {
        "description": "近N日研报评级调高次数(分析师修正漂移, 稀疏机构事件, 妙想家族1)。高=近期被上调最多",
        "available_since": AVAILABLE_SINCE["analyst"],
        "params": {"days": {"type": "int", "min": 1, "max": 20, "default": 5}},
        "compute": _rating_upgrade,
    },
    "moneyflow_surge": {
        "description": "主力净流入突变: main_net 相对自身近N日历史的 z-score(因果)。捕捉「净流入突然放大」的冲击分量, "
                       "区别于 main_net_flow 水平(已证伪=追高马甲)。广发金工口径=突变非水平。"
                       "正=异常流入(疑似机构进场早期), 负=异常流出。仅 2.6 年覆盖(moneyflow 10年回补在跑)",
        "available_since": AVAILABLE_SINCE["moneyflow"],
        "params": {"days": {"type": "int", "min": 5, "max": 60, "default": 20}},
        "compute": _moneyflow_surge,
    },
    "smart_analyst": {
        "description": "聪明钱×分析师确认: 龙虎榜机构净买入强度(滞后1日) × 分析师看多度(3-评分)。"
                       "双确认=机构真金白银买 且 分析师背书。覆盖=dt_inst_net∩分析师, 比 dt_inst_net 更窄。"
                       "广发「聪明钱×分析师」交互: 检验双确认是否放大最强单因子 dt_inst_net",
        "available_since": AVAILABLE_SINCE["analyst"],
        "params": {"days": {"type": "int", "min": 3, "max": 20, "default": 10}},
        "compute": _smart_analyst,
    },
}


def list_factors() -> list[dict]:
    """给 LLM 的数据字典: [{name, description, available_since, params}]。"""
    out = []
    for name, f in FACTORS.items():
        out.append({
            "name": name,
            "description": f["description"],
            "available_since": f["available_since"],
            "params": f["params"],
        })
    return out


def compute_factor(df: pd.DataFrame, name: str, params: dict | None = None) -> pd.Series:
    """按名计算因子。params 覆盖默认参数。"""
    if name not in FACTORS:
        raise KeyError(f"未知因子 '{name}', 可用: {list(FACTORS)}")
    p = dict(FACTORS[name]["params"])
    # 填默认值
    for k, spec in FACTORS[name]["params"].items():
        p[k] = spec.get("default")
    if params:
        p.update(params)
    # 校验参数类型/范围
    for k, spec in FACTORS[name]["params"].items():
        v = p[k]
        if spec["type"] == "int":
            v = int(v)
            if not (spec["min"] <= v <= spec["max"]):
                raise ValueError(f"{name}.{k}={v} 超出范围 [{spec['min']},{spec['max']}]")
            p[k] = v
    return FACTORS[name]["compute"](df, **p)


# ── 组合因子(Phase 2) ───────────────────────────────────────────────────────

def _cross_sectional_zscore(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """当日内横截面 z-score: (x - 当日均值)/当日标准差。保留 df 的 index。

    只用自己的横截面(同日), 不跨日 → 无时间泄露。
    """
    tmp = pd.DataFrame({"date": df["date"].values, "v": s.values}, index=df.index)
    g = tmp.groupby("date")["v"]
    m = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    return (tmp["v"] - m) / sd


def compute_composite(df: pd.DataFrame, components: list[dict], weights: list[int]) -> pd.Series:
    """组合因子 = Σ weight * zscore(component), 权重 ∈ {-1,0,+1}。

    components: [{"factor","params"}, ...]; weights: 等长 int 列表。
    每分量先当日内横截面 z-score(消量纲), 再按权重求和。
    权重 -1 = 做多该因子低值(差分), 0 = 排除该分量。
    """
    if len(components) != len(weights):
        raise ValueError(f"components({len(components)}) 与 weights({len(weights)}) 长度不一致")
    if not any(w != 0 for w in weights):
        raise ValueError("组合至少一个非零权重分量")
    parts = []
    for comp, w in zip(components, weights):
        if w == 0:
            continue
        if w not in (-1, 1):
            raise ValueError(f"权重必须 ∈ {{-1,0,1}}, 得 {w}")
        fv = compute_factor(df, comp["factor"], comp.get("params"))
        parts.append(w * _cross_sectional_zscore(fv, df))
    return sum(parts)


def composite_available_since(components: list[dict]) -> str:
    """组合因子的数据覆盖起始日 = max(各分量 available_since)。

    避免 ret(2年) + moneyflow(6月) 组合被静默截断到 6 个月还不知情。
    """
    return max(FACTORS[c["factor"]]["available_since"] for c in components)


def compute_all_factors(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """返回 DataFrame[factor_name]=因子值(默认参数), index=df.index。用于算相关矩阵。"""
    names = factors or list(FACTORS)
    out = {}
    for n in names:
        p = {k: spec.get("default") for k, spec in FACTORS[n]["params"].items()}
        out[n] = compute_factor(df, n, p)
    return pd.DataFrame(out)


def factor_corr_pairs(df: pd.DataFrame, threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """高相关因子对(冗余提示), 返回 [(f1, f2, r)] 按 |r| 降序。

    相关分量 = 同一信号数两次, 组合它们无增量且虚增显著性, 喂给 LLM 引导避开。
    """
    fdf = compute_all_factors(df)
    c = fdf.corr()
    pairs = []
    cols = list(c.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(c.iloc[i, j])
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(r, 2)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs
