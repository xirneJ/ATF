# runs/ — 实验数据目录

本目录保存**每次训练/评估运行的数据**。以后用 `scripts/launch_run.sh` 启动的运行会自动写入
`runs/<run_name>/`（`config.yaml`、`metrics.jsonl`、`run_summary.json`、`run.log`、
`checkpoint_*`、`images/`），其中大文件已由 `.gitignore` 排除在 git 之外但**仍保存在本目录**。

## 目录说明

| 目录 | 来源 | 说明 |
|---|---|---|
| `20261105_smoke_test/` | `smoke_test.py`（2026-11-05） | 接口 / FD-vs-JVP 数值自检（CPU、GPU-TF32、GPU-highest） |
| `20261105_tiny_fd/` | 对话中贴出的 tiny 训练日志 | 原始 workdir `~/qiangchen/logs/tiny`；FD（δ=0.02）训练 160 step |
| `20261105_tiny_jvp/` | 对话中贴出的 tiny 训练日志 | 原始 workdir `~/qiangchen/logs/tiny_jvp`；JVP 对照，同配置 160 step |

> ⚠️ `20261105_tiny_*` 与 `smoke_test` 是**从对话记录重建**的：当时的代码还没有自动记录功能，
> 完整 stdout 也没有 tee 落盘，因此这里保存的是日志里出现的全部数值指标（逐 step），
> 时间/耗时等元信息是按日志时间戳估算的。原始 checkpoint/images 仍在服务器
> `~/qiangchen/logs/tiny{,_jvp}/`，可以按需拷贝进来（见各目录 readme）。

## 以后自动生成

```bash
bash scripts/launch_run.sh <run_name> <config_mode>
# 例：bash scripts/launch_run.sh 20261105_real_fd pMF_B_16_real_debug
```

汇总所有 run → 论文表格：

```bash
python scripts/collect_runs.py --root runs --md docs/runs.md --csv docs/runs.csv
```
