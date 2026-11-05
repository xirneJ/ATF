# 生成质量评估计划（FID / IS）

回答的问题：**"我们的方法（Chebyshev-u + FD）和传统 pMF 相比，生成质量（FID/IS）到底如何？"**
以及"FD 替代 JVP 是否损害生成质量"。

> 诚实预期：FD 替代 JVP 带来的是**工程收益**（编译快 2.7×、显存小、无需 forward-mode AD），
> **不必然提升 FID**；FID 若有提升，应归因于 Chebyshev 参数化 / 训练策略，而不是导数算法本身。
> 因此论文里把"效率收益"和"质量收益"分开写。

## 现状与缺口

- 仓库已有 JAX InceptionV3 + FID/IS 计算（`utils/fid_util.py`），训练/评估会自动调用
  （`config.fid.cache_ref` 指向参考统计量 npz）。
- **缺**：参考统计量文件（`ref_mu/ref_sigma`）；本脚本补上。
- **缺**：传统 pMF 基线（网络接收 t + 直接输出 u + jvp）——已加 `t_cond` 开关，
  默认 `False`（不影响现有训练），基线配置见 `configs/direct_jvp_b2_config.yml`。

## 步骤 1：子集 train/ref 划分 + 参考统计量（~5 分钟）

```bash
cd ~/qiangchen/ATF
git pull

# 10 类子集：1000 张/类 训练 + 300 张/类 作 FID 参考（3000 张）
python scripts/make_subset_split_for_fid.py \
    --src ~/qiangchen/data/imagenet_subset10 \
    --dst ~/qiangchen/data/imagenet_subset10_split \
    --train-per-class 1000 --ref-per-class 300

# 计算参考统计量（JAX InceptionV3）
python scripts/compute_fid_ref.py \
    --data-dir ~/qiangchen/data/imagenet_subset10_split/ref \
    --out ~/qiangchen/data/fid/subset10_ref.npz
```

## 步骤 2：先用已有 checkpoint 验证 FID 管线（~5–10 分钟，不训练）

目的：确认"采样 → Inception → FID/IS"链路正确（数值会很差，因为模型只训了 1 epoch，属正常）。

```bash
# 复制 eval 模板并把 load_from 改成已有 checkpoint
cp configs/fid/subset10_eval_cheb_config.yml configs/fid/_tmp_eval.yml
# 编辑 _tmp_eval.yml：load_from: /home/xyds25/qiangchen/ATF/runs/20261106_real_fd/checkpoint_3250
python main.py --workdir=runs/eval_smoke_cheb --config=configs/fid/_tmp_eval.yml
# 结果（FID_*/IS_*）写到 runs/eval_smoke_cheb/ 的记录里
```

> 注意：`20261106_real_fd` 的 checkpoint 是"Chebyshev K=3 + FD"架构，因此用 `*_eval_cheb` 模板。

## 步骤 3：传统 pMF 基线正确性检查（2000 步，~15–20 分钟）

先确认 `t_cond=True + cheb_order=0 + jvp` 的基线能正常训练（和"K=0 且无 t 输入"的
`abl_cheb0` 对比，二者应不同）：

```bash
bash scripts/launch_run.sh chk_direct_jvp direct_jvp_b2
python scripts/compare_logs.py --a runs/chk_direct_jvp --b runs/abl_cheb0
```

## 步骤 4：Pilot 对比（子集，ours vs 传统 pMF，同预算）

```bash
bash scripts/launch_run.sh subset10_pilot_ours   fid/subset10_pilot_b2
bash scripts/launch_run.sh subset10_pilot_direct fid/subset10_pilot_direct_jvp_b2
```

- 每个 run：batch 2、20 epoch、每 5 epoch 算一次 subset FID（3000 张生成 vs 3000 张参考）。
- 共享 5090 上约 **6–15 小时/次**（取决于 GPU 占用）；可按时间把 `num_epochs` 调小，两边保持一致。
- 结果直接看 `runs/subset10_pilot_*/metrics.jsonl` 里的 `FID_*` / `IS_*` 字段。
- **这是 pilot**：subset FID 只能做"相对"对比（同协议、同参考集），**不能**和论文里
  ImageNet-256 的 FID 数字直接比。

也可以单独对某个 checkpoint 做 CFG 扫描评估（`ω ∈ {1,2,4,8.5}`、`interval=[0.1,0.7]`）：

```bash
# 编辑 configs/fid/subset10_eval_cheb_config.yml 的 load_from，然后
python main.py --workdir=runs/eval_subset10_cheb --config=configs/load_config.py:fid/subset10_eval_cheb
python main.py --workdir=runs/eval_subset10_direct --config=configs/load_config.py:fid/subset10_eval_direct
```

## 步骤 5：Full-scale（论文级，需要租 8×H100 / TPU）

协议必须两边完全一致（同数据、同 batch、同步数、同 LR/EMA/采样设置）：

1. 全量 ImageNet-256 参考统计量（50k 真实图，50 张/类）：
   ```bash
   python scripts/compute_fid_ref.py \
       --data-dir /mnt/win/data/imagenet/train \
       --per-class 50 --num-samples 50000 \
       --out $HOME/qiangchen/data/fid/imagenet256_ref.npz
   ```
2. 训练两个 run（我们的方法 vs 传统 pMF），配置需把
   `dataset.root=/mnt/win/data/imagenet`、`num_classes=1000`、
   `batch_size=1024`、`num_epochs≈320`、`fid.cache_ref=imagenet256_ref.npz`、
   `fid.num_samples=50000`，其余一致。
3. 评估：用 `configs/fid/subset10_eval_*` 的模板改 `num_classes=1000`、参考统计量换全量，
   CFG 扫描后取最优，比较 FID/IS。

成本粗估：8×H100 约 **¥1.5–3 万 / 3–5 天**（按两家云报价区间）。

## 注意事项

- **类别顺序**：subset 的 train/ref 都由 ImageFolder 按字母序编号，训练与生成用同一
  `num_classes=10` 映射，直接可比；不要混用不同排序的目录。
- **参考样本数**：3000 张参考 + 3000 张生成，在 2048 维特征上协方差估计有偏，FID 噪声较大，
  只用于相对比较；全量实验用 50k 参考才与论文数字可比。
- **CFG 扫描**：1-step 模型对 `ω` / guidance interval 敏感，评估一定要扫描后报最优。
- **随机性**：pilot 若要下结论，建议每个方法跑 2 个 seed，报告均值±标准差。
- 所有 run 的记录（config/metrics/run_summary/README/metadata）自动落在 `ATF/runs/<run>/`。

## 其他可做的生成侧实验（按性价比排序）

1. **多 seed 的 FD vs JVP**（补强"差异在随机波动内"的统计）；
2. **one-step vs 多步采样**（`sampling.num_steps` 2/4/8）看质量-步数曲线；
3. **EMA 教师**（目前未实现）作为后续 idea 的对照；
4. 质量指标扩展：precision/recall、Inception Score（已在 FID 评估里）、CFG 曲线。
