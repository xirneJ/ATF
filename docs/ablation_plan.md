# 消融实验计划（Ablation Plan）

目标：回答论文审稿人会问的四个问题——
1. **有限差分（FD）对 δ 是否敏感？**
2. **中心差分是否必要？**
3. **Chebyshev 多项式基（阶数 K）是否必要 / 几阶合适？**
4. **系数正则是否有助于稳定（抑制 `du_dt_abs` 增长）？**

## 配置矩阵

基线（已有，6500 step）：`runs/20261106_real_fd_b2`（K=3、δ=0.02、中心差分、reg=0）。
对照（已有）：`runs/20261106_real_jvp_b2`（精确 JVP）、`runs/20261106_real_fd_b2_seed43`（随机基线）。

| run 名 | 配置文件 | 变化 | 考察 |
|---|---|---|---|
| `abl_delta001` | `configs/ablation/delta001_config.yml` | `fd_delta=0.01` | δ 敏感性（小） |
| `abl_delta005` | `configs/ablation/delta005_config.yml` | `fd_delta=0.05` | δ 敏感性（大） |
| `abl_forward` | `configs/ablation/forward_config.yml` | `fd_central=False` | 中心差分 vs 前向差分 |
| `abl_cheb0` | `configs/ablation/cheb0_config.yml` | `cheb_order=0` | 无时间基（= 直接输出 u 场） |
| `abl_cheb1` | `configs/ablation/cheb1_config.yml` | `cheb_order=1` | 线性基 |
| `abl_cheb2` | `configs/ablation/cheb2_config.yml` | `cheb_order=2` | 二次基 |
| `abl_reg1e3` | `configs/ablation/reg1e3_config.yml` | `reg_lambda=1e-3` | 系数正则 |

所有消融配置与基线**完全一致**（B/16、ImageNet 10 类子集、batch 2、256²、seed=42、
Muon lr=1e-3、clip=1.0、无辅助损失），只有表中那一项不同，并统一
`training.max_steps: 2000` 控制成本。

## 运行

```bash
cd ~/qiangchen/ATF
git pull
bash scripts/run_ablations.sh
# 或者单独跑某一个：
# bash scripts/launch_run.sh abl_delta001 ablation/δ001   # 注意用 ablation/<name>

# 汇总（截断到前 2000 条记录，与基线可比）
python scripts/ablation_summary.py --root runs --steps 2000 --md docs/ablation_table.md --csv docs/ablation_table.csv
```

预计：7 个 run × 约 15–20 分钟 ≈ **2 小时**（共享 5090、batch 2）。

## 结果解读指南

| 轴 | 好的结果 | 不好的结果 / 结论 |
|---|---|---|
| δ ∈ {0.01, 0.02, 0.05} | 三条曲线接近、`du_dt_abs` 同量级 | 若 δ=0.05 明显偏离 → FD 对 δ 敏感，需给出 δ 选择依据（TF32 下 0.02~0.03） |
| 中心 vs 前向 | 中心差分 `loss_vel` 更低/更稳 | 若前向也差不多 → 可以说明 FD 的鲁棒性；若前向明显差 → 证明中心差分必要 |
| K=0/1/2/3 | 随 K 增大略好、K=3 饱和 | 若 K=0 明显差 → Chebyshev 时间基确实有用（论文卖点）；若差别很小 → 说明瓶颈在别处 |
| reg=0 vs 1e-3 | reg 抑制 `du_dt_abs/loss_reg` 增长且 `loss_vel` 不明显变差 | 若 reg 伤性能 → 说明默认不开正则合理 |

注意：这些消融是 **loss 层面**的（batch 2、10 类子集、2000 step），噪声较大；
判断时应看"平坦均值/趋势"，不要抠单点差异（`--smooth` 可用在 compare_logs 上）。

### ⚠️ 跨 K 比较时不要看 `du_dt_abs`

`du_dt_abs` 的**量级本身依赖参数化方式**，不能跨 Chebyshev 阶数比较：

- K≥1：u 显式含 `T_k(2t-1)`，FD 的 `du/dt` 包含基函数随 t 的变化 → 早期 `du_dt_abs` 就很大；
- K=0：u 无显式 t 依赖，`du/dt` 只来自沿流 z/h 的隐式变化 → 早期极小、后期才追上。

实测 `abl_cheb0` vs 基线（K=3）：step50 的 `du_dt_abs` 相差约两个量级（0.002 vs 0.13），
这属于**结构性差异**而非训练好坏。正确做法：

- 跨 K 的消融（cheb0/1/2）**只比较 `loss_vel`**；
- `du_dt_abs` 只在同 K 的不同设置（δ、中心/前向）之间比较；
- `ablation_summary.py` 输出里的 `mean_du_dt_abs` 列对 cheb0 行请忽略。


## 记录位置

- 每个 run：`ATF/runs/abl_<axis>/`（`config.yaml`、`metrics.jsonl`、
  `run_summary.json`、`README.md`、`metadata.json`、checkpoint/images）
- 汇总表：`docs/ablation_table.md` / `.csv`
- 论文写作时把汇总表 + 关键结论补进 `docs/experiments_history.md`
