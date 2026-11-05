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


def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    # Save only one copy from device 0.
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    if hasattr(checkpoints, "save_checkpoint_multiprocess"):
        checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    else:
        # Newer flax versions removed the *_multiprocess helper; in this code
        # every process holds the same replicated params, so each process can
        # write with overwrite=True (only process 0 should be doing this).
        if jax.process_index() == 0:
            checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)
    log_for_0("Checkpoint step %d saved.", step)
