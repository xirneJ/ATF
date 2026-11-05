# 数据文件说明（Data Dictionary）

本文件说明 `runs/<run_name>/` 下每个文件的含义，以及各字段的定义。
**每个 run 目录里也会自动生成同样的说明**（`README.md` + `metadata.json`，由
`utils/logging_util.Writer` 在运行开始时写入）。

## 目录结构

```
runs/<run_name>/
├── README.md         # 本 run 的说明（自动生成）
├── metadata.json     # 机器可读的说明（自动生成）
├── config.yaml       # 完整配置快照（ml_collections config 的 to_dict）
├── metrics.jsonl     # 逐日志 step 的指标（一行一个 JSON）
├── run_summary.json  # 运行总结（状态/耗时/git/jax 环境/最后指标/完整 config）
├── run.log           # 完整 stdout（用 scripts/launch_run.sh 启动时）
├── checkpoint_*      # 模型 checkpoint（保留个数由 training.checkpoint_keep 控制）
└── images/           # 采样图（use_wandb=False 时，每个 sample_per_epoch 一张）
```

## `metrics.jsonl` 字段

每行是一个 JSON 对象，对应一次 `log_per_step` 的日志边界。

| 字段 | 含义 |
|---|---|
| `step` | 全局训练步（1-based，日志边界） |
| `epoch` | epoch 序号 |
| `loss` | 总训练 loss（经 `adp_wt` 自归一化，量级 ≈1，不能直接看收敛） |
| `loss_vel` | **看收敛用这个**：`sum((V - v_g)^2)` 的 batch 均值（未归一化） |
| `u_abs` | 预测的平均速度 `u` 的平均绝对值 |
| `du_dt_abs` | 有限差分（或 jvp）得到的 `du/dt` 平均绝对值 |
| `loss_reg` | 高阶 Chebyshev 系数正则项（`reg_lambda=0` 时为 0，但该值仍会统计） |
| `lr` | 当前学习率 |
| `steps_per_second` | 相邻日志边界之间的训练吞吐 |
| `aux_loss_lpips` | LPIPS 感知辅助损失（关闭时为 0） |
| `aux_loss_convnext` | ConvNeXt 特征辅助损失（关闭时为 0） |

## `run_summary.json` 字段

| 字段 | 含义 |
|---|---|
| `status` | `running`（每 log step 刷新）或 `completed`（正常结束） |
| `step` | 最后一次记录的 step |
| `started_at` / `updated_at` / `duration_sec` | 时间信息 |
| `hostname` / `workdir` | 运行主机与目录 |
| `git_commit` | 运行时代码的 git commit（写论文复现用） |
| `jax_version` / `jax_backend` / `jax_devices` / `process_count` | JAX 环境 |
| `last_metrics` | 最后一次日志边界的指标（同 `metrics.jsonl` 的一行） |
| `config` | 完整配置（等价于 `config.yaml`） |
| `reconstructed` / `reconstruction_note` | 仅历史重建数据有：说明数据来源与已知近似 |

## `metadata.json` / `README.md` 字段

`metadata.json` 是运行说明的机器可读版本：

| 字段 | 含义 |
|---|---|
| `run_name` / `workdir` / `created_at` | 基本信息 |
| `description` | 这个 run 是什么 |
| `model_str` / `deriv_mode` / `cheb_order` / `fd_delta` / `fd_central` | 方法配置 |
| `dataset` | `synthetic` / `root` / `image_size` / `num_classes` |
| `files` | 每个文件的说明 |
| `metrics_columns` | `metrics.jsonl` 每个字段的说明 |

## `smoke_results.json`（20261105_smoke_test）

| 字段 | 含义 |
|---|---|
| `runs.cpu_fp32` / `runs.gpu_tf32` / `runs.gpu_highest_precision` | 三种精度下的 FD-vs-JVP 误差 |
| `best_delta` / `best_rel_err` / `sweep` | 最优 δ、最优相对误差、δ 扫描结果 |
| `u_norm` / `jvp_du_norm` | 测试点处 u 与 du/dt 的模长 |
| `conclusion` | 结论 |

## `config.yaml`

就是 `configs/default.py` + 对应 yml 合并后的完整配置（dataset/model/training/
sampling/fid/logging/load_from 等），见 `configs/default.py` 里的注释。
