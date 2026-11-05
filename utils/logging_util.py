import logging as _logging
import datetime
import json
import time, os
import shutil

import jax
from absl import logging
import yaml

import numpy as np
from PIL import Image
import wandb


# Description of every file a run writes into its workdir (also written to
# <workdir>/README.md and <workdir>/metadata.json by Writer).
RUN_FILES = {
    "config.yaml": "complete config snapshot used by this run",
    "metrics.jsonl": "one JSON object per logging step (see metrics_columns below)",
    "run_summary.json": "status / step / duration / git commit / jax env / last metrics / full config",
    "checkpoint_*": "model checkpoints (count controlled by training.checkpoint_keep)",
    "images/": "sampled images (only when wandb is disabled)",
    "run.log": "full stdout of the run (when launched via scripts/launch_run.sh)",
    "README.md": "this file: what this run is and what each file means",
    "metadata.json": "machine-readable version of this description",
}

METRICS_COLUMNS = {
    "step": "global training step (1-based at logging boundary)",
    "epoch": "training epoch index",
    "loss": "total training loss (adp_wt-normalized; ~1 by construction)",
    "loss_vel": "raw per-sample sum(V - v_g)^2 mean (NOT adp_wt-normalized)",
    "u_abs": "mean |u| of the predicted average velocity",
    "du_dt_abs": "mean |du/dt| of the finite-difference (or jvp) derivative",
    "loss_reg": "higher-order Chebyshev coefficient regularization term (0 if disabled)",
    "lr": "learning rate at this step",
    "steps_per_second": "training throughput between logging boundaries",
    "aux_loss_lpips": "LPIPS auxiliary loss (0 if disabled)",
    "aux_loss_convnext": "ConvNeXt feature auxiliary loss (0 if disabled)",
}


def log_for_0(*args):
    if jax.process_index() == 0:
        logging.info(*args, stacklevel=2)


class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True


exclude_files = [
    "orbax/checkpoint/async_checkpointer.py",
    "orbax/checkpoint/abstract_checkpointer.py",
    "orbax/checkpoint/multihost/utils.py",
    "orbax/checkpoint/future.py",
    "orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py",
    "orbax/checkpoint/type_handlers.py",
    "orbax/checkpoint/metadata/checkpoint.py",
    "orbax/checkpoint/metadata/sharding.py",
] + [
    "orbax/checkpoint/checkpointer.py",
    "flax/training/checkpoints.py",
] * jax.process_index()
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f"{self.elapse_with_reset():.2f} s"


class MetricsTracker:
    def __init__(self):
        self._sum = None  # tree of numpy arrays (host)
        self._n = 0  # number of steps accumulated on *this host*

    @staticmethod
    def _mean_over_local_devices(x):
        """
        Bring one leaf to host and average over local device axis if present.
        This avoids keeping per-device values around on host.
        """
        # device_get blocks on the computation that produced x.
        a = np.asarray(jax.device_get(x))
        # Under pmap, metrics often have shape [local_devices, ...].
        # If it's already scalar (0-D), leave unchanged.
        if a.ndim >= 1:  # treat leading axis as local device axis
            a = a.mean(axis=0)
        return a

    def update(self, metrics_step_tree):
        """
        Incorporate one step's metrics (per-replica JAX arrays) into the running sum.
        Call this once per training step.
        """
        local_mean = jax.tree_util.tree_map(self._mean_over_local_devices, metrics_step_tree)
        if self._sum is None:
            self._sum = local_mean
        else:
            self._sum = jax.tree_util.tree_map(lambda s, x: s + x, self._sum, local_mean)
        self._n += 1

    def finalize(self):
        """
        Return global mean over steps, devices, and hosts as a tree of Python floats.
        Resets internal state. Safe to call at any logging boundary.
        """
        if self._n == 0:
            return {}

        out = jax.tree_util.tree_map(
            lambda s: float(np.asarray(s / self._n, dtype=np.float64).mean()),
            self._sum,
        )

        self._sum, self._n = None, 0
        return out


class Writer:
    def __init__(self, config, workdir):
        if jax.process_index() != 0:
            return
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)

        # Machine-readable run records (kept for paper plots / tables).
        self.metrics_path = os.path.join(self.workdir, "metrics.jsonl")
        self.summary_path = os.path.join(self.workdir, "run_summary.json")
        try:
            with open(os.path.join(self.workdir, "config.yaml"), "w") as f:
                yaml.safe_dump(config.to_dict(), f, sort_keys=False)
        except Exception as e:
            log_for_0(f"[warn] failed to save config snapshot: {e}")

        # Self-describing run folder: README.md + metadata.json.
        self._write_run_readme(config)

        kwargs = {}

        self.use_wandb = config.logging.use_wandb

        if self.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity if config.logging.wandb_entity else None,
                notes=config.logging.wandb_notes if config.logging.wandb_notes else None,
                tags=config.logging.wandb_tags if config.logging.wandb_tags else None,
                dir="/tmp",  # avoid writing to workdir
                settings=wandb.Settings(_service_wait=60),
                **kwargs,
            )
            wandb.config.update(config.to_dict(), allow_val_change=True)
        else:
            log_for_0("Wandb logging is disabled. Images will be saved to disk.")

    def _write_run_readme(self, config):
        """Write <workdir>/README.md + metadata.json describing this run."""
        try:
            run_name = os.path.basename(os.path.normpath(self.workdir))
            created = datetime.datetime.now().isoformat()

            def get(d, key, default=""):
                try:
                    return d.get(key, default)
                except Exception:
                    return default

            model = get(config, "model", {})
            dataset = get(config, "dataset", {})
            meta = {
                "run_name": run_name,
                "workdir": os.path.abspath(self.workdir),
                "created_at": created,
                "description": (
                    "Training/evaluation run of pixel MeanFlow with Chebyshev-"
                    "parameterized average velocity u and finite-difference "
                    "(or jvp) derivative."
                ),
                "model_str": get(model, "model_str"),
                "deriv_mode": get(model, "deriv_mode"),
                "cheb_order": get(model, "cheb_order"),
                "fd_delta": get(model, "fd_delta"),
                "fd_central": get(model, "fd_central"),
                "dataset": {
                    "synthetic": get(dataset, "synthetic", False),
                    "root": get(dataset, "root"),
                    "image_size": get(dataset, "image_size"),
                    "num_classes": get(dataset, "num_classes"),
                },
                "files": RUN_FILES,
                "metrics_columns": METRICS_COLUMNS,
                "source": "auto-generated by utils/logging_util.Writer",
            }
            with open(os.path.join(self.workdir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)

            lines = [
                f"# Run: `{run_name}`",
                "",
                f"- created_at: `{created}`",
                "- description: training/evaluation run of pixel MeanFlow with "
                "Chebyshev-parameterized average velocity and finite-difference "
                "(or `jax.jvp`) derivative",
                f"- model: `{meta['model_str']}` | deriv_mode: `{meta['deriv_mode']}` | "
                f"cheb_order: `{meta['cheb_order']}` | fd_delta: `{meta['fd_delta']}` "
                f"(central={meta['fd_central']})",
                f"- data: synthetic={meta['dataset']['synthetic']} | "
                f"root=`{meta['dataset']['root']}` | "
                f"image_size={meta['dataset']['image_size']} | "
                f"num_classes={meta['dataset']['num_classes']}",
                "",
                "## Files",
                "",
            ]
            for name, desc in RUN_FILES.items():
                lines.append(f"- `{name}`: {desc}")
            lines += ["", "## metrics.jsonl columns", ""]
            for name, desc in METRICS_COLUMNS.items():
                lines.append(f"- `{name}`: {desc}")
            lines += [
                "",
                "> This file and `metadata.json` are generated automatically by "
                "`utils/logging_util.Writer`; see `docs/run_records.md` for details.",
                "",
            ]
            with open(os.path.join(self.workdir, "README.md"), "w") as f:
                f.write("\n".join(lines))
        except Exception as e:
            log_for_0(f"[warn] failed to write run README/metadata: {e}")

    def write_scalars(self, step, scalar_dict):
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            log_str += f" {k}={v:.5g}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        # Append a machine-readable record (loss curves for later analysis).
        try:
            record = {"step": int(step)}
            for k, v in scalar_dict.items():
                if isinstance(v, (bool, int, float)):
                    record[k] = float(v) if isinstance(v, float) else v
                else:
                    try:
                        record[k] = float(v)
                    except (TypeError, ValueError):
                        record[k] = str(v)
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log_for_0(f"[warn] failed to append metrics.jsonl: {e}")
        if self.use_wandb:
            wandb.log(scalar_dict, step=step)

    def write_summary(self, summary: dict):
        """Write/refresh `run_summary.json` (config snapshot + status + metrics)."""
        if jax.process_index() != 0:
            return
        try:
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            log_for_0(f"[warn] failed to write run summary: {e}")

    def write_images(self, step, image_dict):
        if jax.process_index() != 0:
            return

        def reduce_arr_func(v):
            if isinstance(v, Image.Image):
                return v
            assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
            assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
            assert v.ndim == 3 and 3 in [
                v.shape[0],
                v.shape[2],
            ], "Invalid image shape {}".format(v.shape)
            if v.shape[0] == 3:
                v = v.transpose((1, 2, 0))
            return Image.fromarray(v)

        if self.use_wandb:    
            wandb.log(
                {k: wandb.Image(reduce_arr_func(v)) for k, v in image_dict.items()},
                step=step,
            )
        else:
            if not os.path.exists(f"{self.workdir}/images/"):
                os.makedirs(f"{self.workdir}/images/")
            for k, v in image_dict.items():
                img = reduce_arr_func(v)
                img.save(f"{self.workdir}/images/{step}_{k}.png")

    def __del__(self):
        # Guard against interpreter-shutdown errors (jax backend may already
        # be torn down when __del__ runs).
        try:
            if jax.process_index() != 0:
                return
            if self.use_wandb:
                wandb.finish()
                shutil.rmtree("/tmp/wandb", ignore_errors=True)
        except Exception:
            pass
