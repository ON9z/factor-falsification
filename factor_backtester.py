"""因子回测器 — 把一条「因子假设」(单因子或组合) 变成诚实的历史回测指标。

输入(一条假设): 因子名+参数(或组合分量+权重) + 方向(做多高值/低值) + 选股比例 + 持有期。
输出: 超额收益/IC/IC_IR/t值/胜率/样本量, 按 train(前60%)/test(后40%) 分列,
      另附: 子区间稳定性 + 置换检验 p 值(组合还附 vs_best_single 增量)。

数据口径铁律(Phase 2 P0): 收盘后信号(t日收盘算因子) → 次日开盘执行(t+1 open 入场),
      持有到 t+h 收盘出场。组合路径复用同一口径。

护栏(对应 [[finagent-event-driven-dead-20260829]] 的反过拟合铁律 + 数据三假alpha):
  1. 时点宇宙: 只算当时可交易的 A 股(剔 ETF/基金 5%/1% 前缀), 退市票按 delist_date 剔除。
  2. 数据覆盖: 因子 available_since 之前的区间不计算(否则 turnover=0 伪影)。
  3. 可执行性: 入场日(next open)若涨停(板别感知)则剔除——买不进的一字/封死涨停不算收益。
  4. 交易日对齐: 前向收益按全局交易日历, 不用 raw shift(停牌断档会错位)。
  5. 无泄露: 因子计算只用过去(因果, 见 factor_library); 横截面 rank 只在自己当日。
  6. 出场可执行(缺陷3a): 出场日跌停(一字/收盘跌停)无法卖出 → 顺延到首个可卖日收盘, 不再假设能卖。
  7. 动态摩擦(缺陷3b+F1): 成交额分层往返成本(小成交额股成本更高), 成本闸门=动态成本后净超额显著为正(net t>=2)。
  8. train/test 时间切分 + IC_IR/t 值 + 子区间稳定性 + 置换检验(防过拟合/多重检验)。
  9. 覆盖率哨兵: 逐日「可交易宇宙中因子值非空占比」平均 < 20% = 覆盖子集因子, 判决降级(不冒充全宇宙 alpha)。
 10. 缺失机制审计(#1): 覆盖子集是否系统性偏倚(市值/行业/ST)按日审计, 偏倚即降级(gate 9)。
     中性化(#2): 可选 neutralize='sector,size' 报行业/市值中性化后风险调整 alpha(报告项, 不硬改判决)。
 11. 数据契约预检(#3): 回测前核对「声明的数据不变量」(时点宇宙/价格健全/因子有限),
     任一违反=数据bug硬闸降级(gate 10)。只查声明规则, 未声明缺陷(覆盖/缺失机制)由 gate8/9 兜底。

口径: 入场=信号日 t 的下一个交易日开盘价 open[t+1](T+1 现实), 出场=第 t+h 交易日收盘 close[t+h]。
      fwd_ret = close[t+h]/open[t+1] - 1。基准=当日全市场等权均值 fwd_ret(等权市场组合)。
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import date

import numpy as np
import pandas as pd

from src.agent import factor_library as fl
from src.analysis.limit_up import limit_pct

DB = r"D:\FinAgentData\database\market_data.db"

# 双边摩擦(往返一次): 印花税 0.05% 卖(2023-08 后, 此前 0.1%) + 佣金 ~0.025%*2=0.05% + 过户费 ~0.002%
# → 单次往返 ≈ 0.10%(现)/0.15%(2023 前)。取 0.15% 覆盖全回测期(2016-2023 印花税 0.1% 更高)。
ROUNDTRIP_COST = 0.0015

# ── 缺陷3b: 成交额分层动态成本(Amihud 2002 非流动性 + 中国 size 解释 45-65% 流动性溢价 NAJEF 2020) ──
# 小成交额(小市值)股票真实往返成本远高于大盘股。模型:
#   cost_i = BASE_COST + ILLIQ_COST * median_amount_day / amount_i, 封顶 COST_CAP
# BASE=0.001(印花税0.05%卖+佣金0.05%+过户费≈0.10%, 无冲击), ILLIQ=0.001(中位成交额 +0.10% 冲击 → 往返~0.20%),
# 极端微盘封顶 0.75%。
# 09-05 修正: 原 BASE=0.002/ILLIQ=0.002/CAP=0.015 是注释算术(≈0.10%)与实际常量(0.2%)对不上的 bug,
# 且把印花税按高估口径误算 → 系统性高估 ~2×。按真实 A 股费用(印花税 0.05% 卖单边)重校。
BASE_COST = 0.001
ILLIQ_COST = 0.001
COST_CAP = 0.0075

# ── 覆盖率哨兵(rule.doc 缺陷5: 人审只看逻辑不看数据量) ──
# 横截面覆盖率 = 每个交易日「可交易宇宙中因子值非空」的占比(逐日再平均)。
# 覆盖 < 20% = 覆盖子集因子(如研报目标价只覆盖大盘/特定行业), 其 PASS 只在子集内成立,
# 不可外推到全宇宙, 必须降级/标注, 否则会拿「覆盖子集内相对超额」冒充全宇宙 alpha。
COVERAGE_MIN = 0.20

# 置换检验: 两阶段, 只有 test |t| 超过此门槛才跑(否则直接标"未达置换门槛")
PERM_T_THRESHOLD = 1.5
N_PERM = 100

# ── A6 搜索级闸门: Deflated Sharpe Ratio(DSR) + 累计候选计数 T ──────────
# 反「多重检验自欺」(Bailey & López de Prado 2014): 在同一宇宙上搜了 N 条假设后,
# 纯噪声因子也会有 E[max t] ≈ sqrt(2 ln N) 的「伪显著」极值(爬榜者最易踩的坑)。
# DSR 把观测 Sharpe 按 N 去膨胀, 只有显著高于「N 次试验的期望极值」才算真技能。
# T 持久化到磁盘跨 session 累计 —— 搜索越久, 门槛越高(诚实搜索的核心)。
# 注: 当前每个 backtest 计 1 次试验; 网格搜索的相关试验理想上应去重(留作后续)。

_TRIAL_COUNT_PATH = os.path.join(os.path.dirname(DB), "factor_trial_count.json")
_DSR_MIN_PROB = 0.95  # DSR >= 0.95 才判「真技能概率足够高」


def _read_trial_count() -> int:
    try:
        with open(_TRIAL_COUNT_PATH, "r", encoding="utf-8") as f:
            return int(json.load(f).get("trial_count", 0))
    except (FileNotFoundError, ValueError, OSError):
        return 0


def _increment_trial_count() -> int:
    n = _read_trial_count() + 1
    try:
        os.makedirs(os.path.dirname(_TRIAL_COUNT_PATH), exist_ok=True)
        with open(_TRIAL_COUNT_PATH, "w", encoding="utf-8") as f:
            json.dump({"trial_count": n}, f)
    except OSError:
        pass
    return n


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态分位数(反 CDF)。Acklam 算法, 自包含无 scipy, 精度 ~1e-9。"""
    # 系数(Peter Acklam, 1999)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < plow:  # 下尾
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:  # 中央区
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))  # 上尾
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _deflated_sharpe(sr: float, n_obs: int, skew: float | None, kurt: float | None,
                     n_trials: int, sr_benchmark: float = 0.0) -> float | None:
    """Deflated Sharpe Ratio(Bailey & López de Prado 2014, eq 8)。

    sr: 每期(per-horizon)Sharpe = 均值/标准差。n_obs = 观测数(下注次数)。
    skew/kurt: 超额序列的第三/第四标准化矩(正态 skew=0, kurt=3)。
    n_trials: 同一宇宙累计搜索的独立假设数(>=1, 越大期望极值越高 → 门槛越严)。
    返回 P(true SR > benchmark) ∈ [0,1]; 数据不足返回 None(判 FAIL)。
    """
    if n_obs < 2 or n_trials < 1 or sr is None or skew is None or kurt is None:
        return None
    sr_var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n_obs - 1.0)
    if sr_var <= 0:
        return None
    if n_trials <= 1:
        expected_max = sr_benchmark  # 首次试验无多重检验膨胀
    else:
        emc = 0.5772156649015329  # Euler-Mascheroni 常数
        expected_max = sr_benchmark + math.sqrt(sr_var) * (
            (1.0 - emc) * _norm_ppf(1.0 - 1.0 / n_trials)
            + emc * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    return _norm_cdf((sr - expected_max) / math.sqrt(sr_var))


def _asof_join_fundamental(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """季频基本面 as-of join(点-in-time): 每 (code,date) 取 avail_date ≤ date 的最新报告指标。

    pandas 3.0 的 merge_asof(by=) 要求 on 列全局单调(跨 code 不满足) → 手动按 code searchsorted。
    fund 列 = code, report_date, avail_date, 及指标; df 需有 date(str YYYY-MM-DD)。
    """
    num_cols = [c for c in fund.columns if c not in ("code", "report_date", "avail_date")]
    if not num_cols:
        return df
    # 预分组: code -> 按 avail_date 升序的报告组
    fg_map = {c: g.sort_values("avail_date") for c, g in fund.groupby("code", sort=False)}
    int_date = df["date"].str.replace("-", "").astype(int).to_numpy()
    out_num = {fc: np.full(len(df), np.nan) for fc in num_cols}
    out_rd = np.full(len(df), None, dtype=object)
    for code, g in df.groupby("code", sort=False):
        fg = fg_map.get(code)
        if fg is None or fg.empty:
            continue
        avail = fg["avail_date"].str.replace("-", "").astype(int).to_numpy()
        pos_idx = g.index.to_numpy()
        idx = np.searchsorted(avail, int_date[pos_idx], side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        pos = pos_idx[valid]
        take = idx[valid]
        vals = fg[num_cols].to_numpy()[take]
        for j, fc in enumerate(num_cols):
            out_num[fc][pos] = vals[:, j]
        out_rd[pos] = fg["report_date"].to_numpy()[take]
    df = df.copy()
    for fc in num_cols:
        df[fc] = out_num[fc]
    df["report_date"] = out_rd
    return df


def _asof_join_analyst(df: pd.DataFrame, ar: pd.DataFrame) -> pd.DataFrame:
    """研报评级 as-of join(点-in-time): 每 (code,date) 取 signal_date ≤ date 的最新研报共识, 前向填充。

    ar 列 = code, signal_date, 及指标列(analyst_score/analyst_tp); df 需有 date(str YYYY-MM-DD)。
    signal_date 已在 _load_qfq 里把 report_date(盘后) 映射到首个 ≥ report_date 的交易日(无 look-ahead)。
    稀疏 → 评级水平/目标价在下一份研报前保持(横截面因子, 同 fundamental as-of 口径)。
    """
    num_cols = [c for c in ar.columns if c not in ("code", "report_date", "signal_date")]
    if not num_cols:
        return df
    amap = {c: g.sort_values("signal_date") for c, g in ar.groupby("code", sort=False)}
    int_date = df["date"].str.replace("-", "").astype(int).to_numpy()
    out = {fc: np.full(len(df), np.nan) for fc in num_cols}
    for code, g in df.groupby("code", sort=False):
        ag = amap.get(code)
        if ag is None or ag.empty:
            continue
        avail = ag["signal_date"].str.replace("-", "").astype(int).to_numpy()
        pos_idx = g.index.to_numpy()
        idx = np.searchsorted(avail, int_date[pos_idx], side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue
        pos = pos_idx[valid]
        take = idx[valid]
        vals = ag[num_cols].to_numpy()[take]
        for j, fc in enumerate(num_cols):
            out[fc][pos] = vals[:, j]
    df = df.copy()
    for fc in num_cols:
        df[fc] = out[fc]
    return df


def _report_avail(rd: str) -> str:
    """报告期 → 法定披露截止后的可用日(点-in-time 铁律, 防用未来财报=泄露)。

    A股法定披露截止: 一季报4/30、半年报8/31、三季报10/31、年报次年4/30。
    信号日 t 只能用「报告期+法定截止」已过去的报告(否则用尚未披露的财报=泄露)。
    """
    if not rd or len(rd) < 10:
        return rd
    y, m = rd[:4], rd[5:7]
    if m == "03":
        return f"{y}-04-30"
    if m == "06":
        return f"{y}-08-31"
    if m == "09":
        return f"{y}-10-31"
    if m == "12":
        return f"{int(y) + 1}-04-30"
    return rd


def _load_qfq(c) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT code,date,open,high,low,close,pre_close,change_pct,turnover,volume,amount,"
        "total_mv,circ_mv,pe_ttm,pb "
        "FROM stock_daily_qfq WHERE close>0 ORDER BY code,date", c)
    # 股票宇宙: 剔 ETF/基金(5/1 前缀)
    df = df[~df["code"].str.startswith(("5", "1"))].reset_index(drop=True)
    # ── |ret|>30% 守卫: 堵 close 序列送转断点污染 ──
    # qfq 前复权在送转/转板日 close 序列跳变(前一日近零→当日正常), 导致 close-based 收益
    # 在该日爆 ±200400%(实测 920227 等北交所/大额送转股)。change_pct 已在 DB 侧置 NULL
    # (fix_qfq_changepct.py), 但 close 序列仍含断点, 任何用 close 算 ret_* 的因子会在断点日
    # 得到假大值。这里把「组内 close 日收益 |r|>30%」的行(送转断点, 非新股首日)的 close 置
    # NaN, 让 ret_* 在断点日变 NaN 而非假值。注: 多日动量(ret_5d/20d)窗口若跨越断点仍会污染,
    # 活因子(dt_inst_net/size/insider_net/analyst_upside)均不读 close 收益, 不受影响。
    _grp = df.groupby("code", sort=False)["close"]
    _prev_close = _grp.shift(1)
    _daily_ret = df["close"] / _prev_close - 1
    _broken = _daily_ret.abs() > 0.30
    _broken &= _prev_close.notna()  # 豁免新股首日(无前收)
    if _broken.any():
        df.loc[_broken, "close"] = np.nan
        print(f"_load_qfq: {int(_broken.sum())} 行 close 送转断点(|ret|>30%) 置 NaN", flush=True)
    # 未复权 close(供 analyst_upside 分母): 目标价 max_price 是未复权, 与 qfq close(前复权)相减
    # 会高估高股息/早期年份上行空间(复权口径错配, 09-04 修复, 见 [[finagent-analyst-upside-coverage]])。
    nom = pd.read_sql_query("SELECT code, date, close AS close_nom FROM stock_daily", c)
    df = df.merge(nom, on=["code", "date"], how="left")
    # 估值: pe_ttm/pb/total_mv/circ_mv 现直接来自 stock_daily_qfq(tushare daily_basic 回补,
    # 4 年 2022-07-21+, 见 backfill_market_cap.py), 取代旧 valuation_baostock_hist(仅 1 年)。
    # ps_ttm/pcf_ncf 仍只有 baostock 源, 继续左连接补上(覆盖 2025-08-01+, 更早 NaN)。
    val = pd.read_sql_query(
        "SELECT symbol AS code, date, ps_ttm, pcf_ncf "
        "FROM valuation_baostock_hist", c)
    df = df.merge(val, on=["code", "date"], how="left")
    # 龙虎榜机构席位净买入额(元): 按 (date,code) 聚合机构专用席位 net, 稀疏(仅龙虎榜股有值,
    # 其余 NaN)。供 dt_inst_net 因子(资金承接/出货信号, 新家族)使用。
    inst = pd.read_sql_query(
        "SELECT date, code, SUM(net) AS inst_net FROM dragon_tiger_seat "
        "WHERE seat_type='机构' GROUP BY date, code", c)
    df = df.merge(inst, on=["code", "date"], how="left")
    # 主力资金流向(占比%, 已按东财口径对账): 主力净流入/超大单/大单。盘后即知, 无 look-ahead
    # (与龙虎榜不同, 无需 shift)。medium/small 混源(新浪=0/东财=真值)不可靠, 不用。
    mf = pd.read_sql_query(
        "SELECT symbol AS code, date, main_net, super_large, large "
        "FROM money_flow_daily", c)
    df = df.merge(mf, on=["code", "date"], how="left")
    # 股东增减持(独立库 insider_trades.db, web_server 持 market_data.db 写锁故分离)。
    # 事件按公告日 ann_date 归入首个 >= 公告日的交易日; 净 signed change_ratio(%):
    # IN增持=+, DE减持=-。盘后公告 → factor 里 shift(1) 口径(与 dt_inst_net 同)。
    insider_db = r"D:\FinAgentData\database\insider_trades.db"
    if os.path.exists(insider_db):
        c.execute(f"ATTACH DATABASE '{insider_db}' AS insider")
        ins = pd.read_sql_query(
            "SELECT code, ann_date, in_de, change_ratio FROM insider.insider_trades "
            "WHERE change_ratio IS NOT NULL", c)
        if len(ins):
            ins["signed"] = np.where(ins["in_de"].astype(str).str.upper() == "IN",
                                     ins["change_ratio"], -ins["change_ratio"])
            trading_dates = np.array(sorted(df["date"].unique()))
            dt_ann = pd.to_datetime(ins["ann_date"]).values
            pos = np.searchsorted(trading_dates.astype("datetime64[ns]"), dt_ann, side="left")
            pos = np.minimum(pos, len(trading_dates) - 1)
            ins["date"] = trading_dates[pos]
            ins_agg = (ins.groupby(["code", "date"], sort=False)["signed"]
                       .sum().rename("insider_net").reset_index())
            df = df.merge(ins_agg, on=["code", "date"], how="left")
    # 基本面指标(fundamental_indicators, 季频): 报告期→法定披露截止(avail_date)后 as-of join。
    # 点-in-time 铁律(同龙虎榜 shift(1) 同源): 信号日 t 只能见「报告期+法定截止」已过去的报告,
    # merge_asof backward 取 avail_date ≤ date 的最新报告。季频稀疏 → 值在季度内重复(横截面因子)。
    fund = pd.read_sql_query(
        "SELECT code, report_date, roe_avg, roe_weighted, net_margin, operating_margin, roa, "
        "revenue_growth, profit_growth, equity_growth, asset_growth, "
        "debt_to_asset, current_ratio, quick_ratio, eps_diluted, bvps, ocfps "
        "FROM fundamental_indicators", c)
    if len(fund):
        fund["avail_date"] = fund["report_date"].map(_report_avail)
        df = _asof_join_fundamental(df, fund)
    # 研报评级/目标价(analyst_rating, tushare report_rc): 稀疏机构事件, 盘后发布 → report_date 映射到
    # 首个 ≥ report_date 的交易日(signal_date, 无 look-ahead), 再 as-of join 前向填充(评级水平/目标价
    # 在下一份研报前保持)。同日多券商取共识均值。稀疏 → 无研报覆盖的股票 analyst_* 为 NaN(不进横截面)。
    ar = pd.read_sql_query(
        "SELECT code, report_date, rating_score, max_price, min_price "
        "FROM analyst_rating", c)
    if len(ar):
        ar = ar.dropna(subset=["rating_score", "max_price"], how="all")
        ar = (ar.groupby(["code", "report_date"], as_index=False)
               .agg(analyst_score=("rating_score", "mean"),
                    analyst_tp=("max_price", "mean"))
               .sort_values(["code", "report_date"]))
        trading_dates = np.array(sorted(df["date"].unique()))
        dt_rep = pd.to_datetime(ar["report_date"]).values
        pos = np.searchsorted(trading_dates.astype("datetime64[ns]"), dt_rep, side="left")
        pos = np.minimum(pos, len(trading_dates) - 1)
        ar["signal_date"] = trading_dates[pos]
        df = _asof_join_analyst(df, ar)
    # 研报评级变化(调高/调低)事件: report_date(盘后) → signal_date(首个≥report_date交易日),
    # 次日开盘可买(factor 的 forward 收益=open[t+1] 天然对齐, 无需 shift)。供 rating_upgrade 因子
    # (analyst revision drift 稀疏机构事件, 同 dt_inst_net 家族)。无事件=0(非缺失)。
    rc = pd.read_sql_query(
        "SELECT code, report_date, rating_change FROM analyst_rating "
        "WHERE rating_change IN ('调高','调低')", c)
    if len(rc):
        trading_dates = np.array(sorted(df["date"].unique()))
        dt_rc = pd.to_datetime(rc["report_date"]).values
        p = np.searchsorted(trading_dates.astype("datetime64[ns]"), dt_rc, side="left")
        p = np.minimum(p, len(trading_dates) - 1)
        rc["date"] = trading_dates[p]
        rc["upgrade"] = (rc["rating_change"] == "调高").astype(int)
        rc["downgrade"] = (rc["rating_change"] == "调低").astype(int)
        rc_agg = (rc.groupby(["code", "date"], as_index=False)
                  .agg(upgrade=("upgrade", "sum"), downgrade=("downgrade", "sum")))
        df = df.merge(rc_agg, on=["code", "date"], how="left")
        df["upgrade"] = df["upgrade"].fillna(0)
        df["downgrade"] = df["downgrade"].fillna(0)
    return df


def _load_st_codes(c) -> set:
    return {r[0] for r in c.execute("SELECT code FROM stocks WHERE name LIKE '%ST%'").fetchall()}


def _load_delist(c) -> dict:
    """code -> delist_date(str); 退市当日及以后剔除。"""
    out = {}
    for code, d in c.execute(
            "SELECT code, delist_date FROM stocks WHERE delist_date IS NOT NULL AND delist_date!=''"):
        if d:
            out[code] = str(d)[:10]
    return out


def _load_industry(c) -> dict:
    """code -> 申万一级行业(industry_sw_l1), 空/None 归 '未知'。供中性化 + 缺失机制审计。"""
    return {r[0]: (r[1] or "未知") for r in c.execute(
        "SELECT code, industry_sw_l1 FROM stocks").fetchall()}


def _global_trading_dates(df: pd.DataFrame) -> list[str]:
    """全局交易日序 = 数据里出现过的全部 date(升序)。"""
    return sorted(df["date"].unique().tolist())


def _add_calendar_aligned_forward(df: pd.DataFrame, horizon: int, entry_lag: int = 1):
    """给 df 加 entry_open/entry_chg/entry_gap 与 fwd_close, 交易日历对齐。

    entry_lag=1(默认): 收盘后信号(t 收盘算因子)→ 次日开盘(t+1)入场, 现有所有因子的口径。
    entry_lag=0: 开盘即知信号(如 gap 低开, open/pre_close 集合竞价 09:25 已定)→ 当日开盘入场。
    entry_gap = 入场日开盘跳空(open/pre_close-1), 开盘即知, 供入场侧可买性判断(避开全日 change_pct 未来函数)。

    用全局 date 位置索引 self-merge: (code,pos) -> (code,pos+entry_lag) 的 open/change_pct/pre_close、
    -> (code,pos+h) 的 close。取未来行要 pos 减偏移, 即 entry["pos"] -= entry_lag、fwd["pos"] -= horizon。
    停牌导致某股某 pos+h 无行 → NaN → 该观测自然剔除, 不误对齐到更远的交易日。
    """
    dates = _global_trading_dates(df)
    pos = {d: i for i, d in enumerate(dates)}
    df = df.copy()
    df["pos"] = df["date"].map(pos)

    entry = df[["code", "pos", "open", "change_pct", "pre_close"]].rename(
        columns={"open": "entry_open", "change_pct": "entry_chg",
                 "pre_close": "entry_pre_close"})
    entry["pos"] = entry["pos"] - entry_lag   # 源行 pos+entry_lag 的 open 落到 pos 上
    fwd = df[["code", "pos", "close"]].rename(columns={"close": "fwd_close"})
    fwd["pos"] = fwd["pos"] - horizon  # 源行 pos+h 的 close 落到 pos 上

    out = df.merge(entry, on=["code", "pos"], how="left")
    out = out.merge(fwd, on=["code", "pos"], how="left")
    out["entry_gap"] = out["entry_open"] / out["entry_pre_close"] - 1.0
    return out


def _sellable_exit_close(df: pd.DataFrame, horizon: int) -> pd.Series:
    """出场日跌停后滞(缺陷3a): 若 pos+h 日跌停(一字/收盘跌停, change_pct <= -(limit-0.3)),
    该日无法卖出, 顺延到其后首个可卖(非跌停)交易日的收盘价; 顺延无尽头(停牌/跌停至数据尾)则 NaN。

    与入场侧 buyable(entry_chg < limit-0.3 判涨停不可买) 镜像对称。同时天然覆盖「pos+h 日停牌」
    情形(该股在 pos+h 无行 → searchsorted 落到下一个交易日)。
    """
    lim = np.where(df["is_st"], 5.0, [limit_pct(cd, 0) for cd in df["code"]])
    sellable = df["change_pct"].fillna(0) > -(lim - 0.3)
    out = np.full(len(df), np.nan)
    for code, g in df.groupby("code", sort=False):
        g = g.sort_values("pos")
        pos = g["pos"].to_numpy()
        close = g["close"].to_numpy()
        sell = sellable.loc[g.index].to_numpy()
        n = len(g)
        # next_sell[i] = 首个 j>=i 且 sell[j] 的索引, 无则 n
        next_sell = np.full(n, n, dtype=int)
        nxt = n
        for i in range(n - 1, -1, -1):
            if sell[i]:
                nxt = i
            next_sell[i] = nxt
        j0 = np.searchsorted(pos, pos + horizon)  # 该股首个交易日 >= 全局 pos+horizon
        valid = j0 < n
        exit_idx = np.where(valid, next_sell[np.minimum(j0, n - 1)], n)
        ok = exit_idx < n
        out[g.index[ok]] = close[exit_idx[ok]]
    return pd.Series(out, index=df.index)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """向量 Spearman 秩相关, NaN 已剔除。"""
    a = pd.Series(a).rank().values
    b = pd.Series(b).rank().values
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _neutralize_cross(g: pd.DataFrame, spec: str) -> pd.Series:
    """横截面中性化(单日 g, 已只含非空 f): 去行业均值 + 去对数市值线性成分, 返回残差。

    #2 中性化报告项: 若 alpha 主要来自行业 tilt 或市值 tilt, 中性化后超额会显著缩水/翻号。
    行业中性=业内去均值(去行业因子); 市值中性=对 log(total_mv) 做一元 OLS 取残差(去市值因子)。
    行业+市值联合中性用 Frisch-Waugh-Lovell(先各自去行业均值, 再对行业残差回归市值残差),
    避免顺序中性在「行业与市值相关」(银行偏大盘/医药偏小盘)时留下交叉残差。
    无市值数据者不扣(保持行业中性值), 避免中性化静默缩样本。
    """
    f = g["f"].astype(float)
    ind = g["industry"].fillna("未知")
    want_sector = "sector" in spec and "industry" in g.columns
    want_size = "size" in spec and "total_mv" in g.columns
    if want_sector and want_size:
        u_f = f - f.groupby(ind).transform("mean")           # f 对行业残差
        logmv = np.log(g["total_mv"].where(g["total_mv"] > 0))
        u_s = logmv - logmv.groupby(ind).transform("mean")   # 市值对行业残差
        valid = u_s.notna()
        if valid.sum() > 10 and u_s.loc[valid].std() > 0:
            x = u_s.loc[valid].values
            y = u_f.loc[valid].values
            b = np.cov(x, y)[0, 1] / np.var(x)
            return u_f - b * u_s.fillna(0.0)
        return u_f
    if want_sector:
        return f - f.groupby(ind).transform("mean")
    if want_size:
        logmv = np.log(g["total_mv"].where(g["total_mv"] > 0))
        valid = logmv.notna() & f.notna()
        if valid.sum() > 10 and logmv.loc[valid].std() > 0:
            x = logmv.loc[valid].values
            y = f.loc[valid].values
            b = np.cov(x, y)[0, 1] / np.var(x)
            a = y.mean() - b * x.mean()
            return f - (a + b * logmv).fillna(0.0)
        return f
    return f


def _missing_audit_day(g: pd.DataFrame, miss: pd.Series) -> dict:
    """单日缺失机制(#1): 缺失 vs 非缺失 在市值/ST/行业上的偏倚。返回单日指标供跨日聚合。

    size_ratio = 缺失股平均市值 / 非缺失股平均市值(1=无偏倚, <1=缺失偏小盘)。
    st_ratio   = 缺失股 ST 占比 / 非缺失股 ST 占比(>1=缺失更偏 ST)。
    industry_top_missing_rate = 缺失集中度: 缺失股中占比最高行业的份额(>0.5=缺失集中单一行业)。
    """
    out = {}
    n_miss = int(miss.sum()); n_pres = int((~miss).sum())
    if n_miss == 0 or n_pres == 0:
        return out  # 无缺失/全缺失 → 无偏倚可算
    if "total_mv" in g.columns:
        logmv = np.log(g["total_mv"].where(g["total_mv"] > 0))
        lm_m = logmv[miss].mean(); lm_p = logmv[~miss].mean()
        if pd.notna(lm_m) and pd.notna(lm_p) and lm_m != lm_p:
            out["size_ratio"] = float(np.exp(lm_m - lm_p))
    if "is_st" in g.columns:
        st_m = float(g.loc[miss, "is_st"].mean()); st_p = float(g.loc[~miss, "is_st"].mean())
        if st_p > 0:
            out["st_ratio"] = st_m / st_p
        elif st_m > 0:
            out["st_ratio"] = 10.0
    if "industry" in g.columns:
        ind = g["industry"].fillna("未知")
        miss_share = ind[miss].value_counts(normalize=True)
        if len(miss_share):
            out["industry_top"] = miss_share.idxmax()
            out["industry_top_missing_rate"] = float(miss_share.max())
    return out


def _aggregate_missing_audit(audit_rows: list) -> dict:
    """跨日聚合缺失机制指标: 逐日指标取均值(市值比/ST比), 行业集中度取均值。"""
    if not audit_rows:
        return {}
    a = pd.DataFrame(audit_rows)
    out = {}
    if "size_ratio" in a:
        sr = a["size_ratio"].dropna()
        if len(sr):
            out["size_ratio_mean"] = float(sr.mean())
            out["size_ratio_p10"] = float(sr.quantile(0.1))
            out["size_ratio_p90"] = float(sr.quantile(0.9))
    if "st_ratio" in a:
        st = a["st_ratio"].dropna()
        if len(st):
            out["st_ratio_mean"] = float(st.mean())
    if "industry_top_missing_rate" in a:
        out["industry_top_missing_rate"] = float(a["industry_top_missing_rate"].mean())
        m = a["industry_top"].mode()
        out["industry_top"] = str(m.iloc[0]) if len(m) else None
    return out


def _missing_mechanism_ok(audit: dict, coverage: float | None) -> tuple[bool, str]:
    """#1 缺失机制闸门(gate 9): 覆盖子集若系统性偏倚(市值/行业/ST), 则 PASS 只在该子集成立, 降级。

    覆盖 >= 80% 时缺失影响可忽略(跳过); 否则任一偏倚超阈值 → 非随机 → FAIL。
    """
    if coverage is not None and coverage >= 0.80:
        return True, "覆盖≥80%缺失可忽略"
    if not audit:
        return True, "无审计数据"
    sr = audit.get("size_ratio_mean")
    if sr is not None and (sr < 0.6 or sr > 1.6):
        return False, f"缺失偏市值({sr:.2f}x)"
    st = audit.get("st_ratio_mean")
    if st is not None and st > 2.0:
        return False, f"缺失偏ST({st:.1f}x)"
    ir = audit.get("industry_top_missing_rate")
    if ir is not None and ir > 0.5:
        return False, f"缺失集中行业({audit.get('industry_top')}:{ir:.0%})"
    return True, "缺失随机"


def _data_contract(df: pd.DataFrame, since: str) -> dict:
    """#3 数据契约预检(AlphaSchema「data contracts + leakage rules」× QuantDinger「manifest」借鉴)。

    回测前声明并核对「数据不可违反的不变量」, 作为第一道防线(非唯一防线——覆盖/缺失机制这类
    「未声明的缺陷」由 gate8/gate9 的测量式审计兜底, 见 coverage_min/coverage_std/missing_audit)。

    硬闸只查「数据 bug」类不变量: 时点宇宙/价格健全/因子有限。
    不查「无前视因果性」——那由 factor_library 因果纪律保证(机械验证需逐日截断重算, 代价过高,
    属「未声明」层)。available_since 是覆盖起点声明而非因果边界: 因子在其前有值≠泄露(如 ret 动量
    的 qfq 数据起点早于声明起点), 故「声明前有值」只作信息项, 不当硬闸。

    入参 df 须已带列: date/delist/close/open/f。返回 {checks, notes, all_pass, note}。
    """
    n = len(df)
    checks = []
    # C1 时点宇宙: 退市股按 delist_date 剔除后无残留(声明 + 核对)
    c1 = bool((df["date"] < df["delist"]).all())
    checks.append({"contract": "时点宇宙(无退市后残留)", "pass": c1,
                   "detail": f"残留 {int((~(df['date'] < df['delist'])).sum())}/{n} 行"})
    fvals = df["f"].to_numpy(dtype=float, na_value=np.nan)
    sub = (df["date"] >= since).to_numpy() & ~np.isnan(fvals)
    # C2 价格健全: 覆盖子集内 close/open 无负值/无 inf(NaN=停牌缺失, 交下游 fwd_ret.notna 过滤)
    close_a = df["close"].to_numpy(dtype=float, na_value=np.nan)
    open_a = df["open"].to_numpy(dtype=float, na_value=np.nan)
    bad_px = int((sub & ((close_a < 0) | (open_a < 0) | np.isinf(close_a) | np.isinf(open_a))).sum())
    checks.append({"contract": "价格健全(无负值/inf)", "pass": bad_px == 0,
                   "detail": f"违规 {bad_px} 行"})
    # C3 因子有限: 覆盖子集内因子值无 inf(NaN=缺失, 由 gate8/9 覆盖审计管)
    bad_fv = int((sub & np.isinf(fvals)).sum())
    checks.append({"contract": "因子值有限(无inf)", "pass": bad_fv == 0,
                   "detail": f"违规 {bad_fv} 行"})
    # 信息项(不硬闸): 因子在声明 available_since 前有值 → 数据起点早于声明起点(声明可能过期,
    # 回测未用满可用历史; 也可能是有意覆盖截断)。属覆盖类观察, 交人判断。
    pre = (df["date"] < since).to_numpy()
    pre_rows = int((~np.isnan(fvals) & pre).sum())
    notes = []
    if pre_rows:
        notes.append(f"因子数据起点早于声明 available_since {pre_rows} 行(声明可能过期, 有更多历史未用)")
    hard = all(c["pass"] for c in checks)
    return {"checks": checks, "notes": notes, "all_pass": hard,
            "note": "硬闸=数据bug(时点宇宙/价格/因子非有限); 覆盖类未声明缺陷由 gate8/9 测量审计兜底"}


def _verdict(res: dict) -> dict:
    """多闸门单判决(AAR「几何平均计分」的教训): 任一闸门不过 → 整体 FAIL。

    避免「只盯单指标(如 test 超额)」的爬榜式自欺。闸门:
      1. test 超额 > 0 且 |t| >= 2(有统计意义)
      2. test 超额 > 双边摩擦(ROUNDTRIP_COST, 扣费后仍正)
      3. train 与 test 超额同号(样本外不反转)
      4. 子区间符号一致率 >= 0.75(非单段运气)
      5. 置换 p < 0.05(非横截面噪声); 未达置换门槛(t<1.5)记 FAIL
      6. (仅组合)vs 最强单分量增量 > 0(组合有增量, 单因子跳过此闸门)
      7. DSR >= 0.95(去膨胀: 观测 Sharpe 需显著高于「累计搜索 T 次后的期望极值」)
      8. 覆盖 >= 20%(覆盖子集因子的 PASS 不可外推; 全样本逐日覆盖率均值)
      9. 缺失机制随机(覆盖子集无系统偏倚: 市值/行业/ST)
     10. 数据契约(时点宇宙/价格健全/因子有限, 任一违反=数据bug)

    返回 {"verdict": PASS|FAIL, "failed_gates": [...], "dsr": float|None,
          "trial_count": int, "gates": [{gate,pass}]}。
    """
    te = res.get("test", {}) or {}
    tr = res.get("train", {}) or {}
    gates = []
    te_ex = te.get("excess_per_horizon", 0.0) or 0.0
    te_t = te.get("excess_t", 0.0) or 0.0
    gates.append(("test超额>0且|t|>=2", te_ex > 0 and abs(te_t) >= 2))
    # 缺陷3b + F1: 成本闸门 = 动态成本后净超额显著为正(net t>=2)。旧版只看 net>0, 会放过
    # 「毛显著(t>=2)但净不显著」的因子(如 analyst_upside 净+0.114% t1.11), 见 [[finagent-net-gate-significance-flaw]]。
    te_net = te.get("excess_net_per_horizon")
    te_net_t = te.get("excess_net_t")
    gates.append(("test动态成本后净超额显著为正(net_t>=2)",
                  te_net is not None and te_net > 0
                  and te_net_t is not None and te_net_t >= 2))
    gates.append(("train/test同号", (te_ex > 0) == ((tr.get("excess_per_horizon", 0.0) or 0.0) > 0)))
    stab = res.get("stability", {}) or {}
    sfc = stab.get("sign_consistent_frac")
    gates.append(("子区间符号一致>=0.75", sfc is not None and sfc >= 0.75))
    pv = res.get("p_value_permuted")
    gates.append(("置换p<0.05", pv is not None and pv < 0.05))
    vs = res.get("vs_best_single")
    if vs is not None:
        gates.append(("vs最强单分量有增量", (vs.get("increment", 0.0) or 0.0) > 0))
    # Gate 7 (A6): DSR 去膨胀 —— 累计搜索 T 次后, 观测 Sharpe 是否仍显著高于噪声期望极值
    te_std = te.get("excess_std")
    sr = (te_ex / te_std) if te_std else None
    n_trials = _read_trial_count()
    dsr = _deflated_sharpe(sr, te.get("n_dates", 0), te.get("excess_skew"),
                           te.get("excess_kurt"), n_trials)
    gates.append((f"DSR>=0.95(去膨胀,T={n_trials})", dsr is not None and dsr >= _DSR_MIN_PROB))
    # Gate 8 (覆盖率哨兵): 覆盖子集因子的 PASS 不可外推
    cov = (res.get("full", {}) or {}).get("coverage")
    gates.append(("覆盖>=20%(非覆盖子集因子)", cov is not None and cov >= COVERAGE_MIN))
    # Gate 9 (#1 缺失机制审计): 覆盖子集系统性偏倚(市值/行业/ST) → 降级
    g9_ok, g9_note = _missing_mechanism_ok(res.get("missing_audit") or {}, cov)
    gates.append(("缺失机制随机(覆盖子集无系统偏倚)", g9_ok))
    # Gate 10 (#3 数据契约): 声明的不变量(时点宇宙/价格健全/因子有限)任一违反=数据bug
    dc = res.get("data_contract") or {}
    gates.append(("数据契约(时点宇宙/价格健全/因子有限)", bool(dc.get("all_pass", True))))
    failed = [name for name, ok in gates if not ok]
    return {"verdict": "PASS" if not failed else "FAIL",
            "failed_gates": failed,
            "dsr": dsr,
            "trial_count": n_trials,
            "gates": [{"gate": n, "pass": ok} for n, ok in gates]}


class FactorBacktester:
    def __init__(self, db_path: str = DB, horizon: int = 5):
        self.db_path = db_path
        self.horizon = horizon
        self._cache = {}

    def _data(self):
        if "df" in self._cache:
            return self._cache
        c = sqlite3.connect(self.db_path); c.execute("PRAGMA busy_timeout=5000")
        df = _load_qfq(c)
        st_codes = _load_st_codes(c)
        delist = _load_delist(c)
        ind = _load_industry(c)
        c.close()
        df["is_st"] = df["code"].isin(st_codes)
        df["delist"] = df["code"].map(delist).fillna("9999-12-31")
        df["industry"] = df["code"].map(ind).fillna("未知")
        df = df[df["date"] < df["delist"]].copy()
        self._cache = {"df": df, "st_codes": st_codes}
        return self._cache

    # ── 共享骨架 ────────────────────────────────────────────────────────────

    def _backtest_series(self, df: pd.DataFrame, fv: pd.Series, since: str,
                         direction: str, selection: float, h: int,
                         sellable_exit: bool = True, dynamic_cost: bool = True,
                         entry_lag: int = 1, neutralize: str | None = None):
        """给定因子值 Series(与 df 同 index), 从对齐前向收益到逐日横截面选股。

        sellable_exit(缺陷3a): 出场日跌停顺延到首个可卖日收盘(默认真实口径, 不再假设跌停日能卖)。
        dynamic_cost(缺陷3b): 成交额分层动态成本(小成交额股成本更高), excess_net = 扣除后净超额。
        neutralize(#2): 'sector,size' 之类, 逐日对因子做行业/市值中性化, 报风险调整后 alpha(报告项)。

        返回 (r, raw, audit, contract):
          r        = DataFrame[date,excess,top_ret,bench_ret,ic,n_sel,n_all,cost_top,excess_net,
                               (+excess_neutral/excess_net_neutral 当 neutralize)], 按 date 升序
          raw      = {date: (sel_mask, fwd_ret_array)} 供置换检验(当日内乱序, 用毛收益测信号强度)
          audit    = 缺失机制聚合 dict(市值/ST/行业偏倚), 供 gate 9
          contract = 数据契约预检 dict(时点宇宙/数据起点/价格健全/因子有限), 供 gate 10
        """
        df = df.copy()
        df["f"] = fv.values
        contract = _data_contract(df, since)  # #3 数据契约预检(在 date>=since 过滤前, 才能查前视)
        df = df[df["date"] >= since].copy()

        df = _add_calendar_aligned_forward(df, h, entry_lag)
        if sellable_exit:
            df["fwd_close"] = _sellable_exit_close(df, h)
        df["fwd_ret"] = df["fwd_close"] / df["entry_open"] - 1.0

        lim = np.where(df["is_st"], 5.0, [limit_pct(cd, 0) for cd in df["code"]])
        if entry_lag == 0:
            # 当日开盘入场: 用开盘跳空(开盘即知)判可买, 不用全日 change_pct(收盘才知=未来函数)。
            # 排除一字跌停(做多低开时错杀过滤器)与一字涨停(开盘已封死买不进)。
            g = df["entry_gap"]
            buyable = g.notna() & (g > -(lim - 0.3)) & (g < (lim - 0.3))
        else:
            buyable = df["entry_chg"].fillna(0) < (lim - 0.3)
        df = df[df["fwd_ret"].notna() & buyable].copy()

        if dynamic_cost:
            amt = df["amount"].where(df["amount"] > 0)
            daily_med = df.groupby("date")["amount"].transform("median")
            df["cost"] = (BASE_COST + ILLIQ_COST * (daily_med / amt)).clip(upper=COST_CAP)
        else:
            df["cost"] = ROUNDTRIP_COST

        rows = []
        audit_rows = []
        raw = {}
        for d, g in df.groupby("date", sort=False):
            if len(g) < 30:
                continue
            n_uni = len(g)  # 可交易宇宙(含因子缺失的股票)
            audit_rows.append(_missing_audit_day(g, g["f"].isna()))  # #1 缺失机制(全宇宙)
            g = g[g["f"].notna()]
            n_cov = len(g)  # 因子值非空的覆盖子集
            if n_cov < 30:
                continue
            cov = n_cov / n_uni  # 横截面覆盖率
            # method="first" 并列取首序: 保证恰好选 selection 比例, 避免离散因子(如 analyst_rating
            # 最小值并列占比 54% > selection)在 method="average" 下平均分位越过边界→每天选空→"无样本"。
            rk = g["f"].rank(pct=True, method="first")
            if direction == "high":
                mask = (rk >= 1.0 - selection).values
            else:
                mask = (rk <= selection).values
            sel = g[mask]
            if len(sel) < 5:
                continue
            top_ret = sel["fwd_ret"].mean()
            bench_ret = g["fwd_ret"].mean()
            cost_top = sel["cost"].mean()
            ic = _spearman(g["f"].values, g["fwd_ret"].values)
            row = {
                "date": d, "excess": top_ret - bench_ret, "top_ret": top_ret,
                "bench_ret": bench_ret, "ic": ic, "n_sel": len(sel), "n_all": n_cov,
                "cov": cov,
                "cost_top": cost_top, "excess_net": (top_ret - cost_top) - bench_ret,
            }
            if neutralize:  # #2 中性化残差选股(同一非空宇宙), 报风险调整后 alpha
                fn = _neutralize_cross(g, neutralize)
                rk_n = fn.rank(pct=True, method="first")
                mask_n = (rk_n >= 1.0 - selection).values if direction == "high" else (rk_n <= selection).values
                sel_n = g[mask_n]
                if len(sel_n) >= 5:
                    top_ret_n = sel_n["fwd_ret"].mean()
                    cost_top_n = sel_n["cost"].mean()
                    row["excess_neutral"] = top_ret_n - bench_ret
                    row["excess_net_neutral"] = (top_ret_n - cost_top_n) - bench_ret
                    row["n_sel_neutral"] = len(sel_n)
            rows.append(row)
            raw[str(d)] = (mask, g["fwd_ret"].values)

        r = pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
        audit = _aggregate_missing_audit(audit_rows)
        return r, raw, audit, contract

    @staticmethod
    def _summ(x: pd.DataFrame) -> dict:
        if len(x) == 0:
            return {}
        excess = x["excess"].values
        ic = x["ic"].dropna().values
        std = excess.std(ddof=1)
        t = excess.mean() / std * np.sqrt(len(excess)) if std > 0 else 0.0
        ic_ir = ic.mean() / ic.std(ddof=1) if len(ic) > 1 and ic.std(ddof=1) > 0 else 0.0
        # 第三/第四标准化矩(正态 skew=0, kurt=3), 供 A6 DSR 去膨胀用
        if std > 0 and len(excess) > 3:
            z = (excess - excess.mean()) / std
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))
        else:
            skew = kurt = None
        out = {
            "n_dates": int(len(x)),
            "n_stock_days": int(x["n_sel"].sum()),
            "excess_per_horizon": float(excess.mean()),
            "excess_t": float(t),
            "excess_std": float(std) if std > 0 else None,
            "excess_skew": skew,
            "excess_kurt": kurt,
            "ic_mean": float(np.nanmean(ic)) if len(ic) else None,
            "ic_ir": float(ic_ir),
            "coverage": float(x["cov"].mean()) if "cov" in x else None,
            "win_rate": float(x["win"].mean()) if "win" in x else None,
            "top_ret": float(x["top_ret"].mean()),
            "bench_ret": float(x["bench_ret"].mean()),
            "cost_top": float(x["cost_top"].mean()) if "cost_top" in x else None,
            "excess_net_per_horizon": float(x["excess_net"].mean()) if "excess_net" in x else None,
        }
        # F1 修复: 净超额显著性 — 净序列 t 值, 供 gate2 判「净显著」而非仅「净>0」
        # (旧版只算净均值, 会放过「毛显著净不显著」因子, 见 [[finagent-net-gate-significance-flaw]])
        if "excess_net" in x:
            en = x["excess_net"].values
            stdn = en.std(ddof=1)
            tn = en.mean() / stdn * np.sqrt(len(en)) if stdn > 0 else 0.0
            out["excess_net_t"] = float(tn)
        # #1 时间覆盖率稳定性: 覆盖均值>20% 但某子区间归零 → 仍有覆盖伪影
        if "cov" in x and len(x) > 1:
            out["coverage_min"] = float(x["cov"].min())
            out["coverage_std"] = float(x["cov"].std(ddof=1))
        # #2 中性化后风险调整 alpha(报告项): 行业/市值中性化残差选股的超额与 t 值
        if "excess_neutral" in x:
            en = x["excess_neutral"].values
            stdn = en.std(ddof=1)
            tn = en.mean() / stdn * np.sqrt(len(en)) if stdn > 0 else 0.0
            out["excess_neutral_per_horizon"] = float(en.mean())
            out["excess_neutral_t"] = float(tn)
            out["excess_net_neutral_per_horizon"] = float(x["excess_net_neutral"].mean())
        return out

    @staticmethod
    def _stability(r: pd.DataFrame, k: int = 4) -> dict:
        """子区间稳定性: 把逐日超额按时间切成 k 段, 报每段均值 + 符号一致性。

        符号一致性 = 与「全样本超额符号」相同的分块占比(1.0=全程同号稳健, 0.25=被单段运气驱动)。
        若某因子只在单个子区间赚钱(其余全亏), 说明是碰运气不是稳健 alpha。
        """
        n = len(r)
        if n < k * 2:
            return {"chunks": [], "sign_consistent_frac": None}
        chunks = []
        for i in range(k):
            c = r.iloc[int(n * i / k):int(n * (i + 1) / k)]
            if len(c):
                chunks.append(round(float(c["excess"].mean()), 5))
        overall_sign = 1 if r["excess"].mean() > 0 else (-1 if r["excess"].mean() < 0 else 0)
        match = sum(1 for x in chunks if (x > 0) == (overall_sign > 0))
        return {"chunks": chunks, "sign_consistent_frac": round(match / k, 2)}

    @staticmethod
    def _permutation_pvalue(raw: dict, n_perm: int = N_PERM) -> float | None:
        """置换检验: 当日内乱序因子标签, 破坏「因子秩→收益」关系但保留收益的时间结构。

        原假设 = 因子无横截面预测力。统计量 = 逐日(top组收益 - 全体均值)的样本均值。
        p = 乱序后 |统计量| >= 观测 |统计量| 的占比。返回 None 表示样本不足。
        """
        if not raw or len(raw) < 20:
            return None
        obs = np.array([float(ret[mask].mean() - ret.mean()) for mask, ret in raw.values()])
        obs_mean = float(obs.mean())
        cnt = 0
        for _ in range(n_perm):
            pm = np.array([float(np.random.permutation(ret)[mask].mean() - ret.mean())
                           for mask, ret in raw.values()])
            if abs(pm.mean()) >= abs(obs_mean):
                cnt += 1
        return round((cnt + 1) / (n_perm + 1), 4)

    def _summarize_rows(self, r: pd.DataFrame, raw: dict, p_value_permuted: float | None = None) -> dict:
        """train/test 切分 + 稳定性 + 置换 p 值, 组装成指标 dict。"""
        if len(r) == 0:
            return {"error": "无足够样本(数据覆盖太短?)"}
        r = r.copy()
        r["win"] = (r["excess"] > 0).astype(int)
        split = int(len(r) * 0.6)
        tr, te = r.iloc[:split], r.iloc[split:]
        out = {
            "full": self._summ(r),
            "train": self._summ(tr),
            "test": self._summ(te),
            "stability": self._stability(r),
        }
        if p_value_permuted is not None:
            out["p_value_permuted"] = p_value_permuted
        return out

    def _finalize(self, r, raw, meta: dict, audit: dict | None = None,
                  contract: dict | None = None) -> dict:
        """组装最终返回 dict: 基础指标 + (可选)置换 p 值 + 缺失机制审计 + 数据契约 + 中性化报告。"""
        res = self._summarize_rows(r, raw)
        res.update(meta)
        # 置换两阶段: test |t| 过门槛才跑
        te = res.get("test", {})
        if te and abs(te.get("excess_t", 0)) >= PERM_T_THRESHOLD:
            res["p_value_permuted"] = self._permutation_pvalue(raw)
        res["net_of_friction"] = round(ROUNDTRIP_COST, 4)  # 静态参考值(缺陷3b 后实际成本见 cost_top)
        res["cost_model"] = {"base": BASE_COST, "illiq": ILLIQ_COST, "cap": COST_CAP}
        # #1 缺失机制审计: 覆盖子集是否系统性偏倚(市值/行业/ST), 供 gate 9 判降级
        res["missing_audit"] = audit or {}
        # #3 数据契约预检: 声明的不变量核对结果, 供 gate 10 判降级(数据bug)
        res["data_contract"] = contract or {}
        # #2 中性化报告: 行业/市值中性化后风险调整 alpha(报告项, 不硬改判决)
        if meta.get("neutralize"):
            tn = res.get("test", {})
            res["neutralized_test_excess"] = tn.get("excess_neutral_per_horizon")
            res["neutralized_test_t"] = tn.get("excess_neutral_t")
        res["verdict"] = _verdict(res)
        return res

    # ── 单因子 ─────────────────────────────────────────────────────────────

    def backtest(self, factor: str, params: dict | None = None, direction: str = "high",
                 selection: float = 0.10, horizon: int | None = None,
                 available_since: str | None = None, entry_lag: int = 1,
                 neutralize: str | None = None) -> dict:
        """跑一条单因子假设。返回指标 dict。

        direction: 'high' 做多因子高值组(动量), 'low' 做多低值组(反转)。
        selection: 选股比例(0~1), 0.10=顶/底 10%。
        neutralize: 逗号分隔 'sector,size' 之类, 报行业/市值中性化后风险调整 alpha(报告项)。
        """
        h = horizon or self.horizon
        trial_index = _increment_trial_count()  # A6: 每条假设 = 一次搜索, 计入累计 T
        data = self._data()
        df = data["df"]
        since = available_since or fl.FACTORS[factor]["available_since"]

        fv = fl.compute_factor(df, factor, params)
        r, raw, audit, contract = self._backtest_series(df, fv, since, direction, selection, h,
                                                        entry_lag=entry_lag, neutralize=neutralize)
        if len(r) == 0:
            return {"factor": factor, "params": params, "error": "无足够样本(数据覆盖太短?)"}
        return self._finalize(r, raw, {
            "factor": factor, "params": params or {}, "direction": direction,
            "selection": selection, "horizon_days": h, "available_since": since,
            "trial_index": trial_index, "neutralize": neutralize,
        }, audit=audit, contract=contract)

    # ── 组合(Phase 2) ─────────────────────────────────────────────────────

    def backtest_composite(self, components: list[dict], weights: list[int],
                           direction: str = "high", selection: float = 0.10,
                           horizon: int | None = None,
                           neutralize: str | None = None) -> dict:
        """跑一条组合假设。components=[{factor,params}], weights ∈ {-1,0,1}。

        额外输出 vs_best_single: 组合 test 超额 - 最强单分量 test 超额(增量), 增量不显著 = 组合无意义。
        neutralize: 同 backtest, 报中性化后风险调整 alpha(报告项)。
        """
        h = horizon or self.horizon
        trial_index = _increment_trial_count()  # A6: 组合本身也是一次搜索
        data = self._data()
        df = data["df"]
        since = fl.composite_available_since(components)

        fv = fl.compute_composite(df, components, weights)
        r, raw, audit, contract = self._backtest_series(df, fv, since, direction, selection, h,
                                                        neutralize=neutralize)
        if len(r) == 0:
            return {"components": components, "weights": weights, "error": "无足够样本(数据覆盖太短?)"}

        res = self._finalize(r, raw, {
            "factor": "composite", "components": components, "weights": weights,
            "direction": direction, "selection": selection, "horizon_days": h,
            "available_since": since, "n_components": sum(1 for w in weights if w != 0),
            "trial_index": trial_index, "neutralize": neutralize,
        }, audit=audit, contract=contract)

        # vs_best_single: 每个分量单独回测(方向 = 权重符号: +1→high, -1→low), 取 test 超额最强
        best = None
        for comp, w in zip(components, weights):
            if w == 0:
                continue
            single = self.backtest(comp["factor"], comp.get("params"),
                                   direction="high" if w == 1 else "low",
                                   selection=selection, horizon=h)
            te = single.get("test", {}).get("excess_per_horizon")
            if te is None:
                continue
            if best is None or te > best["excess"]:
                best = {"factor": comp["factor"], "params": comp.get("params", {}),
                        "direction": "high" if w == 1 else "low", "excess": te}
        if best is not None:
            comp_excess = res.get("test", {}).get("excess_per_horizon", 0.0)
            res["vs_best_single"] = {
                "best_factor": best["factor"], "best_direction": best["direction"],
                "best_test_excess": best["excess"],
                "composite_test_excess": comp_excess,
                "increment": round(comp_excess - best["excess"], 5),
            }
            res["verdict"] = _verdict(res)  # vs_best_single 落地后重算多闸门判决
        return res

    # ── 相关矩阵(供 researcher prompt 引导 LLM 避开冗余组合) ────────────────

    def correlation_pairs(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """7 原子因子(默认参数)的高相关对, 用于喂 LLM。"""
        df = self._data()["df"]
        df = df[df["date"] >= fl.FACTORS["ret"]["available_since"]]
        return fl.factor_corr_pairs(df, threshold)

    def factor_series(self, factor: str, params: dict | None = None) -> pd.Series:
        """给定因子的值序列(与 df 同 index), 供新颖性检验 / 相关分析。"""
        return fl.compute_factor(self._data()["df"], factor, params)

    def default_factor_frame(self) -> pd.DataFrame:
        """默认参数下全部因子的 DataFrame(列=因子名), 供相关矩阵 / 新颖性闸门。"""
        return fl.compute_all_factors(self._data()["df"])


if __name__ == "__main__":
    import json
    bt = FactorBacktester(horizon=5)
    print("== 单因子 smoke ==")
    for name, p, d in [("ret", {"days": 5}, "high"), ("rel_ret", {"days": 20}, "low")]:
        t0 = time.time()
        res = bt.backtest(name, p, direction=d)
        print(f"[{name} {p} {d}] 耗时{time.time()-t0:.1f}s "
              f"test超额={res['test'].get('excess_per_horizon',0):+.4f} "
              f"stability={res.get('stability',{}).get('chunks')}")
    print("\n== 组合 smoke: rel_ret(20)反转 + turnover_zscore(20)低吸 等权 ==")
    t0 = time.time()
    comp = bt.backtest_composite(
        [{"factor": "rel_ret", "params": {"days": 20}},
         {"factor": "turnover_zscore", "params": {"days": 20}}],
        [1, 1], direction="high")
    print(f"耗时{time.time()-t0:.1f}s")
    print(json.dumps(comp, ensure_ascii=False, indent=2, default=str))
