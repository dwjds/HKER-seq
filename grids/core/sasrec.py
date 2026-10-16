
trial_params = dict()

fixed_params = dict(
    batch_size=2048,
    learning_rate=0.001,
    hidden_units=64,
    inner_size=256,  # RecBole: FFN 64->256->64
    num_blocks=2,
    dropout_rate=0.5,  # RecBole 0.5 (was 0.6)
    maxlen=50,

    # ####
    num_heads=2,
    l2_emb=0.0,
    initializer_range=0.02,  # RecBole
    train_last_only=True,   # RecBole: CE only on last position
    # maxlen = None,
    batch_quota=None,
    seed = 0,
    sampler_seed = 789,
    device = None,
    max_epochs = 400,
    # step_replacement = None, # during tests, max value will be replaced with an optimal one
)

