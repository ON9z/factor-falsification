# FinAgent Factor Falsification

> A股（中国股市）横截面 alpha 因子的**证伪记录**：28 个候选因子，在最严的 10 道闸门验证标准下，
> 逐个判生死。结论先行——**0 个因子通过诚实验证。**

这不是一份「我们找到了 alpha」的报告，而是一份「我们证伪了 27 条走不通的路」的报告。
负面结果本身是对交易最值钱的东西：它替你划掉了那些你迟早会去踩的坑。

---

## 结论：0 / 42 存活

**验证矩阵**：14 个因子 × 持有期 h ∈ {5, 10, 20} 交易日 = 42 组（`analyst_rating` 因样本不足
记「无样本」，实际 39 组有结果）。

在 10 道闸门（见 [`validation-standard.md`](validation-standard.md)）的「任一闸门不过即 FAIL」
判据下，**只有 3 组 PASS**：

| 因子 | 方向 | 持有期 | 判决 | 但真相是 |
|---|---|---|---|---|
| `size` | 做多小市值 | h=10 | PASS | **β 不是 α**：幸存者偏差修正后超额转负，是众所周知的小市值溢价 |
| `size` | 做多小市值 | h=20 | PASS | 同上 |
| `analyst_upside` | 做多目标价上行空间 | h=20 | PASS | **复权口径 bug（虚高 ~2×）+ 零售不可执行 + 动量 β**（见下） |

三个 PASS 在**更深的核查**下全部塌掉：

- **`size`**：`size` 是唯一双 regime 稳健的因子，但它是 β（小市值溢价）不是可捕获的 α。
  加入退市/幸存者修正后，超额转负。
- **`analyst_upside`**（研报目标价上行空间）：
  1. **复权口径 bug**：修前 test_t 虚高 ~2×（h20：buggy 10.6 → fixed 7.4）。
  2. **零售不可执行**：目标价只覆盖 ~27% 的股票（大盘/热门股），扣动态成本后净超额不显著
     （test_net_retail t=0.62）。
  3. **动量 β**：按年拆分，超额**全部集中在 2016–2017**（小市值 β 时代：h20 的 2016=+0.121、
     2017=+0.229），**2021–2025 全为负**（2024=−0.035，t=−7.6）。它不是「研报洞察」，
     是「小市值动量」的马甲。

**一句话**：最严标准下活下来的 3 个，要么是 β、要么是 bug、要么是马甲。**净结果 = 0。**

---

## 目录

| 文件 | 内容 |
|---|---|
| [`README.md`](README.md) | 本文件：结论 + 总览 |
| [`validation-standard.md`](validation-standard.md) | 10 道闸门的完整定义 + 防过拟合机器 |
| [`factor-graveyard.md`](factor-graveyard.md) | 每个已证伪因子的「死因」逐条清单 |
| [`factor_backtester.py`](factor_backtester.py) | 验证引擎源码（从 FinAgent 抽取） |
| [`factor_library.py`](factor_library.py) | 28 个原子因子定义源码（从 FinAgent 抽取） |
| `data/reverify_factors.json` | 42 组判决的完整原始数据（逐闸门） |
| `data/combined_rerun_20260904.json` | `analyst_upside` buggy/fixed + 按年拆分 |
| `data/analyst_upside_*.json` | `analyst_upside` 的集中度 / 零售可执行性 |
| `data/bite_*.json` | 覆盖子集漂移 / 机构净流入成员口径 |
| `data/validate_seat_structure_result.json` | 龙虎榜席位结构的基线对比 |

---

## 这 28 个因子是什么

按来源分组（详见 [`factor-graveyard.md`](factor-graveyard.md)）：

| 组 | 因子 | 结果 |
|---|---|---|
| **量价/技术** | ret, rel_ret, turnover_avg, turnover_zscore, vol_ratio, close_pos, ret_vol, gap | 全死 |
| **估值** | pe_ttm, pb, ps_ttm, pcf_ncf | 全死（`ps`/`pcf` 数据仅 1 年，更早证伪） |
| **基本面** | roe, roa, net_margin, revenue_growth, profit_growth, debt_to_asset | 全死（多数 test 超额为负） |
| **研报/分析师** | analyst_rating, analyst_upside, rating_upgrade, smart_analyst | `analyst_upside` 伪 PASS（已塌），余全死 |
| **资金流/席位** | dt_inst_net, insider_net, main_net_flow, super_large_flow, moneyflow_surge | 全死（`dt_inst_net` 机构-only 曾为正，09-04 转负） |
| **风格** | size | β 非 α（幸存者修正后转负） |

> 完整结论背后的历史：日线 GBDT/LSTM AUC≈0.50；分钟模型 0.588 永远猜跌（重训泄漏 0.711→0.510）；
> 估值 PE/PB/PS/PCF 全死；21 个技术因子全死；资金流/龙虎榜席位/涨停派生/读博弈全死；
> 网格超额 100% = 半仓暴露不是收割器。这些都在 `factor-graveyard.md` 里有据可查。

---

## 为什么要开源这个

1. **负面结果难发、更难见光**，但它们是防止别人（和自己）重造轮子的最有效护栏。
2. **验证标准比因子本身更值钱**：10 道闸门 + Deflated Sharpe + 动态成本 + 覆盖率哨兵，
   这套「防自欺机器」可以直接拿去审你自己的因子。
3. **诚实是 quant 唯一能长期持有的仓位**。这份仓库的立场是：把「我们试过、失败了」说清楚，
   比假装「我们找到了圣杯」更有长期价值。

## 数据与复现

- **数据源**：A 股全市场前复权日线（`stock_daily_qfq`，2016-01 起 10 年）+ tushare/巨潮/新浪的
  估值、基本面、研报、资金流、龙虎榜。数据本体**不含在本仓库**（版权+体积），只含**判决结果**。
- **口径**（`factor_backtester.py` 顶部有完整注释）：
  - 入场 = 信号日 t 的**下一个交易日开盘价** `open[t+1]`（T+1 现实，无未来函数）
  - 出场 = 第 t+h 交易日收盘 `close[t+h]`
  - `fwd_ret = close[t+h]/open[t+1] − 1`，基准 = 当日全市场等权均值
  - 一字/封死涨停买不进 → 剔除；出场跌停卖不出 → 顺延到首个可卖日
  - 退市票按 `delist_date` 剔除（时点宇宙，无幸存者偏差）
- **代码依赖**：`factor_backtester.py` / `factor_library.py` 抽取自私有项目 FinAgent，
  依赖 `pandas`/`numpy` + 一个本地 SQLite 市场数据库；**不能脱离 FinAgent 独立运行**，
  开源的是「标准」和「结论」，不是可一键跑通的独立回测器（数据不在仓库里）。

## 立场声明

本仓库是个人研究记录，不是投资建议。A 股交易须遵守中国法律法规。所有因子均为横截面
统计检验，不构成任何买卖信号。
