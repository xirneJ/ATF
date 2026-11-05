"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()

    dataset.root = ""

    dataset.num_workers = 8
    dataset.prefetch_factor = 4
    dataset.pin_memory = False

    # Debugging only: use a deterministic random dataset instead of ImageNet.
    dataset.synthetic = False
    dataset.synthetic_samples = 1024

    dataset.image_size = 256
    dataset.image_channels = 3
    dataset.num_classes = 1000
    dataset.use_flip = False

    # ------------------------------------------------------------
    # Training
    config.training = training = ml_collections.ConfigDict()

    training.learning_rate = 0.0001
    training.batch_size = 256

    training.num_epochs = 1000

    training.log_per_step = 100
    training.sample_per_epoch = 10
    training.checkpoint_per_epoch = 10
    training.fid_per_epoch = 10
    training.half_precision = False

    training.seed = 42

    training.adam_b2 = 0.95
    training.ema_val = [0.9999]

    training.grad_clip_norm = 1.0  # global grad-norm clip before the optimizer; <=0 disables

    # Stop after this many optimizer steps (0 = run full num_epochs).
    # Useful for cheap ablation runs.
    training.max_steps = 0

    # Number of recent checkpoints to keep. Set to e.g. 1000 (or 0) for paper
    # runs where you want to keep every checkpoint; keep small to save disk.
    training.checkpoint_keep = 3

    training.lr_schedule = "warmup_const"
    training.warmup_epochs = 0

    # ------------------------------------------------------------
    # MeanFlow
    config.model = model = ml_collections.ConfigDict()
    model.num_classes = dataset.num_classes

    # Noise Distribution
    model.P_mean = -0.4
    model.P_std = 1.0

    # Loss
    model.data_proportion = 0.5
    model.cfg_beta = 1.0
    model.cfg_max = 7.0
    model.class_dropout_prob = 0.1

    # Training Dynamics
    model.norm_p = 1.0
    model.norm_eps = 0.01

    # Chebyshev parameterization of the average velocity u and derivative
    # estimation (finite difference along the flow by default)
    model.cheb_order = 3
    model.t_cond = False  # True + cheb_order=0 + deriv_mode='jvp' = traditional pMF baseline
    model.deriv_mode = "fd"  # "fd" | "jvp"
    model.fd_delta = 1e-3
    model.fd_central = False
    model.reg_lambda = 0.0  # weight for higher-order coefficient regularization
    model.teacher_ema = False  # reserved; not implemented yet

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_steps = 1
    sampling.num_classes = dataset.num_classes

    # ------------------------------------------------------------
    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.device_batch_size = 40
    fid.cache_ref = ""

    # ------------------------------------------------------------
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.wandb_project = ""
    logging.wandb_entity = ""
    logging.wandb_notes = ""
    logging.wandb_tags = []

    # others
    config.load_from = ""
    config.eval_only = False

    return config
