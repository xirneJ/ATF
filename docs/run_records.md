# 运行记录（Run Records）

为了方便写论文，训练/评估的**每次运行**都会在 `--workdir` 目录下自动保存以下文件。
**约定：`--workdir` 默认放在仓库内的 `runs/<run_name>/`**（用 `scripts/launch_run.sh`
启动会自动这样设置），这样一个项目的所有数据都在 `ATF/runs/` 下，方便查看。

```
runs/<run_name>/
├── README.md         # 本 run 的说明（自动生成：这是什么 run、每个文件是什么、列的含义）
├── metadata.json     # 机器可读的说明（自动生成）
├── config.yaml       # 运行时的完整配置快照（dataset/model/training/sampling/...）
├── metrics.jsonl     # 逐日志 step 的机器可读指标（loss / loss_vel / du_dt_abs / u_abs ...）
├── run_summary.json  # 运行总结：status、step、耗时、git commit、jax 版本/设备、
│                     # hostname、最后一次指标、完整 config（便于写论文表格）
├── run.log           # launch_run.sh 保存的完整 stdout
├── checkpoint_*      # 模型 checkpoint（保留个数由 training.checkpoint_keep 控制）
└── images/           # 每个 sample_per_epoch 生成的采样图（use_wandb=False 时）
```

> 每个 run 目录里都会自动生成 `README.md` + `metadata.json`，逐个文件/逐列说明数据含义；
> 字段级定义见 [`docs/data_dictionary.md`](data_dictionary.md)。

> `.gitignore` 已把 `runs/**/checkpoint*`、`runs/**/images/`、`run.log` 排除在 git 之外：
> 文件仍然保存在 `ATF/runs/` 里，但不会把几十 GB 的 checkpoint 提交进仓库。

## 推荐启动方式

```bash
bash scripts/launch_run.sh 20261105_real_fd pMF_B_16_real_debug
# 等价于：
#   python main.py --workdir=<repo>/runs/20261105_real_fd \
#       --config=configs/load_config.py:pMF_B_16_real_debug 2>&1 | tee runs/20261105_real_fd/run.log
```

历史实验记录（2026-11-05 的 smoke/tiny FD/tiny JVP 数据）见
[`docs/experiments_history.md`](experiments_history.md)。

## 说明

- `run_summary.json` 在训练过程中**每 `log_per_step` 步刷新一次**，所以即使中途
  Ctrl-C / 崩溃，也会留下截至最后一次日志点的记录（status 为 `running`）；
  正常结束时写成 `completed`。
- `metrics.jsonl` 是追加写入的，可直接用 pandas / 脚本画 loss 曲线：
  ```python
  import pandas as pd
  df = pd.read_json("<workdir>/metrics.jsonl", lines=True)
  ```
- `checkpoint_keep` 默认 3（只留最近 3 个）。论文阶段需要保留全部 checkpoint 时，
  在配置里设为 `training.checkpoint_keep: 0`（或 1000），注意磁盘占用
  （B/16 一个 checkpoint 约 2–3GB）。

## 汇总所有运行

```bash
# 打印所有 run 的 markdown 表格；顺便落盘成文件
python scripts/collect_runs.py --root $HOME/qiangchen/logs \
    --md docs/runs.md --csv docs/runs.csv
```

## 建议

1. **每次运行都用 `tee` 保存完整 stdout**（run_summary 之外的双保险）：
   ```bash
   python main.py --workdir=... 2>&1 | tee $HOME/qiangchen/logs/<run>.log
   ```
2. 长训练/正式实验建议打开 `logging.use_wandb: True`，曲线自动存云端。
3. workdir 起名要能读出实验含义，例如：
   `logs/20261105_pmfB16_fd_d0.02_b8_sub10/`。
4. 竞价/临时实例：先备份 `workdir/` 和日志再跑。
