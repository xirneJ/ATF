# 20261105_tiny_jvp — tiny 模型 JVP 对照训练（历史数据，重建）

- **来源**：2026-11-05 对话中贴出的终端日志；原始 workdir `~/qiangchen/logs/tiny_jvp`
- **目的**：作为 `20261105_tiny_fd` 的对照，`deriv_mode="jvp"`（精确导数），其他配置完全相同
- **状态**：completed，160 step（5 epoch × 32 step）
- **配置**：见 `run_summary.json`；编译时间 sample 16.8s / `p_train_step` 85.8s
- **数据文件**：
  - `metrics.jsonl` — 每 5 step 一行的指标（从日志重建）
  - `run_summary.json` — 配置 / 最终指标 / 编译时间（重建）
- **关键结果**：与 FD 版本逐 step 对比，`loss_vel` 最大相对差约 **1e-4**（几乎完全重合）
  → 证明 FD 替代 JVP 在训练动力学上等价。
- **原始 checkpoint / images**：仍在服务器 `~/qiangchen/logs/tiny_jvp/`，可拷贝过来（大文件已被 gitignore）：
  ```bash
  cp -r ~/qiangchen/logs/tiny_jvp/checkpoint_* runs/20261105_tiny_jvp/
  cp -r ~/qiangchen/logs/tiny_jvp/images        runs/20261105_tiny_jvp/
  ```
