import optuna
from grids.core import hypsasrec, generate_base_config


def generate_config(trial: optuna.Trial) -> dict:
    trial_params = hypsasrec.trial_params
    fixed_params = dict(
        **hypsasrec.fixed_params,
        # overwrite c values if set;
        # based on https://canyon-indigo-809.notion.site/b1a23bafebc846a4b2c87d73c7a4b943
        c = 1.0, # obtained from SVD embeddings at 1e^{-5} tolerance
        # 可选：训练结束后保存权重，供案例分析使用。例如:
        checkpoint_dir = './data/checkpoints/hypsasrec_mooccube',
    )
    suggest = trial.suggest_categorical
    config = generate_base_config(trial_params, fixed_params, suggest)
    return config