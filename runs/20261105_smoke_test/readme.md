# 20261105_smoke_test — 接口 / FD 数值自检（历史数据，重建）

- **来源**：2026-11-05 对话中贴出的 `smoke_test.py` 输出
- **目的**：验证重构后的代码能跑通，并量化"有限差分 du/dt"与"`jax.jvp` 精确导数"的误差
- **模型**：`pmfDiT_tiny`（32×32，patch 8，hidden 64，depth 4），`cheb_order=3`
- **数据文件**：`smoke_results.json`
  - CPU(fp32)：最佳 δ=3e-3，相对误差 4.9e-5
  - GPU(默认 TF32)：最佳 δ=3e-2，相对误差 5.7e-3（δ 变小误差反而变大 → TF32 舍入）
  - GPU(`JAX_DEFAULT_MATMUL_PRECISION=highest`)：最佳 δ=3e-3，误差 5.0e-5（与 CPU 一致）
- **结论**：FD 实现正确；GPU 默认 TF32 时需把 δ 调到 ~0.02，或使用高精度矩阵乘。
