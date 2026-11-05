import jax
from flax.training import checkpoints

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    state = checkpoints.restore_checkpoint(workdir, state)
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def save_checkpoint(state, workdir, keep=3):
    """
    Saves the model state to a checkpoint in the specified working directory.

    Args:
        keep: number of recent checkpoints to keep. Use a large value (e.g.
            1000) or 0/negative to keep (almost) all checkpoints for paper runs.
    """
    # Save only one copy from device 0.
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    keep = 3 if keep is None else int(keep)
    if keep <= 0:
        keep = 10**6  # effectively keep all
    log_for_0("Saving checkpoint step %d (keep=%d).", step, keep)
    if hasattr(checkpoints, "save_checkpoint_multiprocess"):
        checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=keep)
    else:
        # Newer flax versions removed the *_multiprocess helper; in this code
        # every process holds the same replicated params, so each process can
        # write with overwrite=True (only process 0 should be doing this).
        if jax.process_index() == 0:
            checkpoints.save_checkpoint(workdir, state, step, keep=keep, overwrite=True)
    log_for_0("Checkpoint step %d saved.", step)
