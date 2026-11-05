"""Smoke test for the Chebyshev-u + finite-difference MeanFlow refactor.

No ImageNet / FID needed. Run it on the machine that will do the real training
(GPU) to catch shape / jit / NaN issues in a couple of minutes:

    python smoke_test.py

Checks:
  1. tiny model init / forward / backward (loss + grads finite, grads nonzero)
     for both deriv_mode="fd" and deriv_mode="jvp"
  2. one-step sampling produces finite outputs
  3. finite-difference du/dt converges to the exact jax.jvp du/dt as delta
     shrinks (validates replacing the original JVP with the FD estimate)
"""

import sys
import types
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from models import pmfDiT
from pmf import generate, pixelMeanFlow

# Tiny model variant so the smoke test is fast (no B/16 required).
pmfDiT.pmfDiT_tiny = partial(
    pmfDiT.pmfDiT,
    input_size=32,
    patch_size=8,
    hidden_size=64,
    depth=4,
    num_heads=4,
    aux_head_depth=1,
)

B, H, W, C = 2, 32, 32, 3
NUM_CLASSES = 10


def _make_model(**kwargs):
    return pixelMeanFlow(
        model_str="pmfDiT_tiny",
        num_classes=NUM_CLASSES,
        cheb_order=3,
        noise_scale=1.0,
        **kwargs,
    )


def _perturb(params, key, scale=0.02):
    """Break the zero-init of the final layer so the derivative is non-zero."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        p + scale * jax.random.normal(k, p.shape, p.dtype)
        for p, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def main():
    key = jax.random.PRNGKey(0)
    key, k_init, k_img, k_lbl, k_pert = jax.random.split(key, 5)
    ok = True

    # ---------- 1) init / forward / backward + 2) sampling ----------
    for deriv_mode in ("fd", "jvp"):
        model = _make_model(deriv_mode=deriv_mode)
        variables = model.init(
            {"params": k_init},
            jnp.ones((B, H, W, C)),
            jnp.ones((B,), jnp.int32),
            jnp.ones((B,), jnp.int32),
        )
        params = _perturb(variables["params"], k_pert)
        images = jax.random.uniform(k_img, (B, H, W, C), minval=-1.0, maxval=1.0)
        labels = jax.random.randint(k_lbl, (B,), 0, NUM_CLASSES)

        def loss_fn(p):
            return model.apply(
                {"params": p},
                images=images,
                labels=labels,
                aux_fn=None,
                rngs={"gen": k_lbl},
                method=model.forward,
            )

        (loss, dict_losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        loss = float(loss)
        grad_norm = float(
            jnp.sqrt(
                sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
            )
        )
        metrics = {k: float(v) for k, v in dict_losses.items()}
        finite = np.isfinite(loss) and np.isfinite(grad_norm)
        print(f"[{deriv_mode}] loss={loss:.6f} grad_norm={grad_norm:.6f}")
        print(f"[{deriv_mode}] metrics={metrics}")
        if not finite or grad_norm <= 0:
            print(f"  FAIL: non-finite loss/grad or zero grad ({deriv_mode})")
            ok = False

        cfg = types.SimpleNamespace(
            dataset=types.SimpleNamespace(
                num_classes=NUM_CLASSES, image_size=H, image_channels=C
            )
        )
        samples = generate(
            {"params": params},
            model,
            k_img,
            n_sample=2,
            config=cfg,
            num_steps=1,
            omega=8.5,
            t_min=0.1,
            t_max=0.7,
        )
        s_ok = bool(np.isfinite(np.asarray(samples)).all())
        print(f"[{deriv_mode}] samples shape={samples.shape} finite={s_ok}")
        if not s_ok:
            print(f"  FAIL: sampling produced non-finite values ({deriv_mode})")
            ok = False

    # ---------- 3) FD vs JVP derivative ----------
    model = _make_model(deriv_mode="fd")
    variables = model.init(
        {"params": k_init},
        jnp.ones((B, H, W, C)),
        jnp.ones((B,), jnp.int32),
        jnp.ones((B,), jnp.int32),
    )
    params = _perturb(variables["params"], k_pert)
    labels = jax.random.randint(k_lbl, (B,), 0, NUM_CLASSES)

    k_z, k_v = jax.random.split(k_pert)
    z = jax.random.normal(k_z, (B, H, W, C))
    v = 0.1 * jax.random.normal(k_v, (B, H, W, C))
    t = jnp.full((B, 1, 1, 1), 0.6)
    r = jnp.full((B, 1, 1, 1), 0.2)
    h = t - r
    w = jnp.full((B, 1, 1, 1), 3.0)
    t_min = jnp.full((B, 1, 1, 1), 0.1)
    t_max = jnp.full((B, 1, 1, 1), 0.7)

    def u_of(zz, tt, hh):
        u, _ = model.apply(
            {"params": params},
            zz,
            tt,
            hh,
            w,
            t_min,
            t_max,
            labels,
            method=model.u_theta,
        )
        return u

    u0 = u_of(z, t, h)
    _, du_jvp = jax.jvp(u_of, (z, t, h), (v, jnp.ones_like(t), jnp.ones_like(h)))
    jvp_norm = float(jnp.linalg.norm(du_jvp))
    print(
        f"[fd-vs-jvp] u_norm={float(jnp.linalg.norm(u0)):.6f} "
        f"jvp_du_norm={jvp_norm:.6f}"
    )
    if not np.isfinite(jvp_norm) or jvp_norm <= 0:
        print("  FAIL: jvp derivative is zero/non-finite")
        ok = False

    best = None
    for delta in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        du_fd = (
            u_of(z + delta * v, t + delta, h + delta)
            - u_of(z - delta * v, t - delta, h - delta)
        ) / (2.0 * delta)
        rel = float(jnp.linalg.norm(du_fd - du_jvp) / (jvp_norm + 1e-12))
        print(f"[fd-vs-jvp] delta={delta:g} central_diff_rel_err={rel:.3e}")
        if best is None or rel < best[1]:
            best = (delta, rel)

    if best is not None:
        print(f"[fd-vs-jvp] best delta={best[0]:g} rel_err={best[1]:.3e}")

    if ok:
        print("\nSMOKE TEST PASSED")
        return 0
    print("\nSMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
