import flax.linen as nn
import jax
import jax.numpy as jnp

from models import pmfDiT


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    sample_idx=None,
):
    """
    Generate samples from the model

    Args:
        variable: Model parameters.
        model: pixelMeanFlow model.
        rng: JAX random key.
        n_sample: Number of samples to generate.
        config: Configuration object.
        num_steps: Number of sampling steps.
        omega: CFG scale.
        t_min, t_max: Guidance interval.
        sample_idx: Optional index for class-conditional sampling.

    Returns:
        images: Generated images.
    """
    num_classes = config.dataset.num_classes
    img_size, img_channels = config.dataset.image_size, config.dataset.image_channels

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng, rng_xt, rng_sample = jax.random.split(rng, 3)

    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype) * model.noise_scale

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

    def step_fn(i, x_i):
        return model.apply(
            variable,
            x_i,
            y,
            i,
            t_steps,
            omega,
            t_min,
            t_max,
            method=model.sample_one_step,
        )

    images = jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    return images


class pixelMeanFlow(nn.Module):
    """pixel MeanFlow with a Chebyshev-parameterized average velocity.

    The network predicts the average velocity as

        u(z, t, h) = sum_{k=0}^{cheb_order} c_k(z, h, ...) * T_k(2t - 1),
        h = t - r,

    where T_k are Chebyshev polynomials of the first kind. The instantaneous
    velocity is recovered from the MeanFlow identity

        v = u + (t - r) * du/dt,

    with du/dt estimated by a finite difference along the flow (default) or by
    `jax.jvp`. The training target v_g (CFG-guided instantaneous velocity) is
    unchanged from the original pMF; the teacher is the u head evaluated at
    h = 0 (u(z, t, t) = v(z, t)).
    """

    # Model and dataset
    model_str: str
    dtype = jnp.float32
    num_classes: int = 1000

    # Noise distribution
    P_mean: float = -0.4
    P_std: float = 1.0
    cfg_max: float = 7.0
    noise_scale: float = 1.0

    # Loss
    data_proportion: float = 0.5
    cfg_beta: float = 1.0
    class_dropout_prob: float = 0.1

    # Training dynamics
    norm_p: float = 1.0
    norm_eps: float = 0.01

    # Chebyshev parameterization of u and derivative estimation
    cheb_order: int = 3
    deriv_mode: str = "fd"  # "fd" | "jvp"
    fd_delta: float = 1e-3
    fd_central: bool = False
    reg_lambda: float = 0.0
    teacher_ema: bool = False

    # perceptual
    lpips: bool = False
    lpips_lambda: float = 1.0
    convnext: bool = False
    convnext_lambda: float = 0.0
    perceptual_max_t: float = 1.0

    tr_uniform: float = False

    # Evaluation mode
    eval: bool = False

    def setup(self):
        """
        Setup pixel MeanFlow model.
        """
        if self.teacher_ema:
            raise NotImplementedError(
                "teacher_ema is not implemented yet. Keep it False and use the "
                "shared u head at h=0 as the teacher."
            )
        net_fn = getattr(pmfDiT, self.model_str)
        self.net: pmfDiT.pmfDiT = net_fn(
            name="net",
            num_classes=self.num_classes,
            num_basis=self.cheb_order + 1,
            eval=self.eval,
        )

    #######################################################
    #               Chebyshev parameterization            #
    #######################################################

    def _cheb_basis(self, t):
        """Chebyshev T_k(2t-1), k=0..cheb_order, each of shape (B, 1, 1, 1)."""
        x = 2.0 * t - 1.0
        basis = [jnp.ones_like(x)]
        if self.cheb_order >= 1:
            basis.append(x)
        for k in range(2, self.cheb_order + 1):
            basis.append(2.0 * x * basis[k - 1] - basis[k - 2])
        return basis

    def u_theta(self, x, t, h, w, t_min, t_max, y):
        """Average velocity from the Chebyshev-parameterized network.

        Args:
            x: noisy image z_t, shape (B, H, W, C)
            t, h, w, t_min, t_max: conditioning scalars, broadcastable to (B,)
            y: class labels, shape (B,)

        Returns:
            u: (B, H, W, C) average velocity
            coeffs: (B, H, W, cheb_order + 1, C) Chebyshev coefficients
        """
        bz = x.shape[0]
        t_flat = t.reshape(bz)
        h_flat = h.reshape(bz)
        w_flat = w.reshape(bz)
        t_min_flat = t_min.reshape(bz)
        t_max_flat = t_max.reshape(bz)

        coeffs = self.net(x, t_flat, h_flat, w_flat, t_min_flat, t_max_flat, y)

        k1 = self.cheb_order + 1
        channels = coeffs.shape[-1] // k1
        coeffs = coeffs.reshape(
            bz, coeffs.shape[1], coeffs.shape[2], k1, channels
        )

        basis = self._cheb_basis(t_flat.reshape(bz, 1, 1, 1))
        u = jnp.zeros(
            (bz, coeffs.shape[1], coeffs.shape[2], channels), dtype=coeffs.dtype
        )
        for k in range(k1):
            u = u + coeffs[:, :, :, k, :] * basis[k]

        return u, coeffs

    #######################################################
    #                       Solver                        #
    #######################################################

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
        """
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        omega = jnp.broadcast_to(omega, (bsz,))
        t_min = jnp.broadcast_to(t_min, (bsz,))
        t_max = jnp.broadcast_to(t_max, (bsz,))

        u, _ = self.u_theta(z_t, t, t - r, omega, t_min, t_max, y=labels)

        return z_t - jnp.einsum("n,n...->n...", t - r, u)

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        """
        Sample t and r from logit-normal distribution.
        """
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)

        if self.tr_uniform:
            # 10% random tr samples
            unif_mask = jax.random.uniform(
                self.make_rng("gen"), (bz, 1, 1, 1), dtype=self.dtype
            ) < 0.1
            t = jnp.where(unif_mask, jax.random.uniform(
                self.make_rng("gen"), (bz, 1, 1, 1), dtype=self.dtype
            ), t)
            r = jnp.where(unif_mask, jax.random.uniform(
                self.make_rng("gen"), (bz, 1, 1, 1), dtype=self.dtype
            ), r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        return t, r, fm_mask

    def sample_cfg_scale(self, bz, s_max=7.0):
        """
        Sample CFG scale omega from power distribution.
        """
        ukey = self.make_rng("gen")
        u = jax.random.uniform(
            ukey, (bz, 1, 1, 1), minval=0.0, maxval=1.0, dtype=jnp.float32
        )

        if self.cfg_beta == 1.0:  # special case for \int 1/x
            s = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            smax = jnp.asarray(s_max, jnp.float32)
            b = jnp.asarray(self.cfg_beta, jnp.float32)

            log_base = (1.0 - b) * jnp.log1p(smax)
            log_inner = jnp.log1p(u * jnp.expm1(log_base))

            s = jnp.exp(log_inner / (1.0 - b))

        return jnp.asarray(s, jnp.float32)

    def sample_cfg_interval(self, bz, fm_mask=None):
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        """
        rng_start, rng_end = jax.random.split(self.make_rng("gen"))

        t_min = jax.random.uniform(
            rng_start, (bz, 1, 1, 1), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (bz, 1, 1, 1), minval=0.5, maxval=1.0, dtype=self.dtype
        )

        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

        return t_min, t_max

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def v_cond_fn(self, x, t, omega, y):
        """
        Instantaneous velocity from the u head at h = 0 (i.e. u(z, t, t) = v).

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v: Predicted instantaneous velocity.
        """

        # h = 0, t_min / t_max are dummy values for the instantaneous limit
        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)

        v, _ = self.u_theta(x, t, h, omega, t_min, t_max, y=y)

        return v

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned instantaneous velocities.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v_c: Predicted v conditioned on class labels.
            v_u: Predicted v without class labels.
        """
        bz = x.shape[0]

        # Create duplicated batch for conditioned and unconditioned predictions
        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.full((bz,), self.num_classes, dtype=y.dtype)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        w = jnp.concatenate([omega, jnp.ones_like(omega)], axis=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]

        rand_mask = (
            jax.random.uniform(self.make_rng("gen"), shape=(bz,))
            < self.class_dropout_prob
        )
        num_drop = jnp.sum(rand_mask).astype(jnp.int32)
        drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

        labels = jnp.where(
            drop_mask.reshape(bz),
            self.num_classes,
            labels,
        )
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, used as the
                 tangent for the derivative estimate.
        """

        # compute CFG target
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)

        v_c = self.v_cond_fn(z_t, t, w, y=y)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = jnp.where(fm_mask, v_g_fm, v_g)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, labels, aux_fn=None):
        """
        Forward process of pixel MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).

        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        t, r, fm_mask = self.sample_tr(bz)
        h = t - r

        rng = self.make_rng("gen")
        rng, rng_used1 = jax.random.split(rng)
        rng, rng_used2 = jax.random.split(rng)

        e = jax.random.normal(rng_used1, x.shape, dtype=self.dtype) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / jnp.clip(t.reshape((-1, 1, 1, 1)), 0.05, 1.0)

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self.sample_cfg_scale(bz, s_max=self.cfg_max)

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
        )

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # ---- average velocity at (z_t, t, h) ----
        u1, coeffs1 = self.u_theta(z_t, t, h, omega, t_min, t_max, labels)

        # ---- total derivative du/dt along the flow ----
        # tangent: (dz/dt, dt/dt, dr/dt) = (v_c, 1, 0)  =>  dh/dt = 1
        if self.deriv_mode == "fd":
            dt = jnp.full_like(h, self.fd_delta)
            dz = self.fd_delta * v_c
            u_p, _ = self.u_theta(
                z_t + dz, t + dt, h + dt, omega, t_min, t_max, labels
            )
            if self.fd_central:
                u_m, _ = self.u_theta(
                    z_t - dz, t - dt, h - dt, omega, t_min, t_max, labels
                )
                du_dt = (u_p - u_m) / (2.0 * self.fd_delta)
            else:
                du_dt = (u_p - u1) / self.fd_delta
        elif self.deriv_mode == "jvp":
            dtdt = jnp.ones_like(t)
            dhdh = jnp.ones_like(h)

            def _u_fn(zz, tt, hh):
                return self.u_theta(zz, tt, hh, omega, t_min, t_max, labels)

            (u1, coeffs1), (du_dt, _) = jax.jvp(
                _u_fn, (z_t, t, h), (v_c, dtdt, dhdh)
            )
        else:
            raise ValueError(f"Unknown deriv_mode: {self.deriv_mode}")

        # Our compound function V = u + (t - r) * du/dt
        V = u1 + h * jax.lax.stop_gradient(du_dt)

        v_g = jax.lax.stop_gradient(v_g)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # pixel MeanFlow objective is conceptually v-loss
        loss_vel = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_vel = adp_wt_fn(loss_vel)

        # Regularize the higher-order Chebyshev coefficients (not via adp_wt_fn)
        if self.cheb_order >= 2:
            reg = jnp.zeros((bz,), dtype=self.dtype)
            for k in range(2, self.cheb_order + 1):
                reg = reg + jnp.mean(coeffs1[:, :, :, k, :] ** 2, axis=(1, 2, 3))
        else:
            reg = jnp.zeros((bz,), dtype=self.dtype)
        loss_reg = jnp.mean(reg)

        # aux loss
        if self.convnext or self.lpips:
            assert aux_fn is not None, "auxiliary loss function is not provided."

            pred_x = z_t - t * u1

            aux_loss_lpips, aux_loss_convnext = aux_fn(
                pred_x, x, rng_used2
            )  # shape (B,)

            mask = t.flatten() < self.perceptual_max_t

            aux_loss_lpips = jnp.where(mask, aux_loss_lpips, 0.0)
            aux_loss_convnext = jnp.where(mask, aux_loss_convnext, 0.0)

            aux_loss = (
                adp_wt_fn(aux_loss_lpips) * self.lpips_lambda
                + adp_wt_fn(aux_loss_convnext) * self.convnext_lambda
            )
        else:
            aux_loss = aux_loss_lpips = aux_loss_convnext = 0.0

        loss = jnp.mean(loss_vel + self.reg_lambda * reg + aux_loss)

        dict_losses = {
            "loss": loss,
            "loss_vel": jnp.mean((V - v_g) ** 2),
            "loss_reg": loss_reg,
            "aux_loss_lpips": jnp.mean(aux_loss_lpips),
            "aux_loss_convnext": jnp.mean(aux_loss_convnext),
            "u_abs": jnp.mean(jnp.abs(u1)),
            "du_dt_abs": jnp.mean(jnp.abs(du_dt)),
        }

        return loss, dict_losses

    def __call__(self, x, t, y):
        return self.net(x, t, t, t, t, t, y)  # initialization only
