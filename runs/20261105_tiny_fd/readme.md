# 20261105_tiny_fd — tiny 模型 FD 训练（历史数据，重建）

- **来源**：2026-11-05 对话中贴出的终端日志；原始 workdir `~/qiangchen/logs/tiny`
- **目的**：验证"Chebyshev 参数化 u + 有限差分 du/dt"的训练循环（与 `20261105_tiny_jvp` 对照）
- **状态**：completed，160 step（5 epoch × 32 step）
- **配置**：见 `run_summary.json`（`pmfDiT_tiny`、batch 16、synthetic 512、`deriv_mode=fd`、
  `fd_central=True, fd_delta=0.02`、EMA 1 份）
- **数据文件**：
  - `metrics.jsonl` — 每 5 step 一行的指标（从日志重建）
  - `run_summary.json` — 配置 / 最终指标 / 编译时间（重建）
- **关键结果**：`loss_vel` 1.333 → 1.187；`u_abs` 0.0043 → 0.279；`du_dt_abs` 0.0118 → 0.632；
  `loss_reg` 1.2e-5 → 2.4e-2（随机数据下正常，真实数据需关注）。
- **原始 checkpoint / images**：仍在服务器 `~/qiangchen/logs/tiny/`，可拷贝过来（大文件已被 gitignore）：
  ```bash
  cp -r ~/qiangchen/logs/tiny/checkpoint_* runs/20261105_tiny_fd/
  cp -r ~/qiangchen/logs/tiny/images        runs/20261105_tiny_fd/
  ```
