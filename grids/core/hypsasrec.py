from . import sasrec

trial_params = dict(**sasrec.trial_params)

fixed_params = dict(
        **sasrec.fixed_params,
        # c = 1,
        bias = True,
        geom = 'ball',
        # Full HKER-Seq uses SeqHypformer.  The w/o Causal Mask experiment overrides
        # only this flag to False, thereby restoring the original Hypformer.
        use_temporal_mask = True,
)
