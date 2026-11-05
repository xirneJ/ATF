import jax.numpy as jnp


def make_grid_visualization(vis, grid=8, max_bz=8):
    """Arrange a batch of images into a grid (works for any batch size).

    Args:
        vis: (N, H, W, C) images.
        grid: maximum number of columns (and rows) per grid.
        max_bz: maximum number of grids to return.

    Returns:
        (G, row*H, col*W, C) array, G = min(number_of_grids, max_bz).
    """
    assert vis.ndim == 4
    n, h, w, c = vis.shape
    if n == 0:
        return vis

    col = max(1, min(int(grid), n))
    row = max(1, min(int(grid), n // col))
    per_grid = col * row

    n_grids = (n + per_grid - 1) // per_grid
    n_use = n_grids * per_grid
    if n < n_use:
        pad = jnp.zeros((n_use - n, h, w, c), dtype=vis.dtype)
        vis = jnp.concatenate([vis, pad], axis=0)
    else:
        vis = vis[:n_use]

    # Layout: each grid has `col` columns; each column is `row` consecutive images.
    vis = vis.reshape((n_grids, col, row, h, w, c))
    vis = jnp.transpose(vis, (0, 2, 3, 1, 4, 5))  # (grids, row, h, col, w, c)
    vis = vis.reshape((n_grids, row * h, col * w, c))

    return vis[: min(max_bz, n_grids)]
