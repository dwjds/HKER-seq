import argparse
from . import defaults


class ArgparserException(Exception): pass


def parse_args(is_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='hypsasrec', type=str)                                                       #修改！！   
    parser.add_argument('--dataset',default='mooccube', type=str)                                                        #修改！！   
    parser.add_argument('--time_offset', default=defaults.time_offset_q, type=float,
                        help='Used only when --split_strategy=time (global time quantile for train/test).')
    parser.add_argument('--split_strategy', choices=('time', 'recbole_ls'), default='recbole_ls', type=str,
                        help="'time': split by time_offset; 'recbole_ls': per-user leave-one-out (order=TO, group_by=user, LS valid_and_test).")
    parser.add_argument('--target_metric', choices=('HR', 'MRR', 'NDCG'), default=defaults.target_metric, type=str)
    parser.add_argument('--topn', default=10, type=int)
    parser.add_argument(
        '--report_topn',
        nargs='+',
        type=int,
        default=None,
        help='Additional top-N cutoffs evaluated by the same trained model.'
    )
    parser.add_argument('--config_path', default='./grids/hypsasrec_mooccube.py', type=str)                             #修改！！
    parser.add_argument('--exhaustive', default=False, action="store_true")
    parser.add_argument('--grid_steps', default=1, type=int) # 0 means run infinitely, None will use defaults
    parser.add_argument('--check_best', default=False, action="store_true")
    parser.add_argument('--save_config', default=False, action="store_true")
    parser.add_argument('--dump_results', default=False, action="store_true")
    parser.add_argument(
        '--config_overrides',
        default=None,
        type=str,
        help='JSON object merged into the selected grid config, e.g. \'{"c": 0.5}\'.'
    )
    parser.add_argument(
        '--result_path',
        default=None,
        type=str,
        help='Optional JSON path for final test metrics. Requires --check_best.'
    )
    parser.add_argument(
        '--resume_checkpoint',
        default=None,
        type=str,
        help='Checkpoint path used to automatically save and resume training.'
    )
    parser.add_argument('--es_tol', default=0.001, type=float)
    parser.add_argument('--es_max_steps', default=10, type=int)
    parser.add_argument('--next_item_only', default=False, action="store_true")
    parser.add_argument(
        '--no_final_refit_on_train_plus_valid',
        default=False,
        action='store_true',
        help='Evaluate the selected model on test after retraining on train only, '
             'so all compared methods use the same train/valid/test protocol.'
    )
    # saving/resuming studies via RDB:
    parser.add_argument('--study_name', default=None, type=str)
    parser.add_argument('--storage', choices=('sqlite', 'redis'), default=None, type=str)
    args = parser.parse_args()
    validate_args(args, is_test)
    return args


def validate_args(args, is_test):
    if not is_test and not args.config_path:
        # models that require hyper-params tuning must be provided with config file
        # models without hyper-parameters are only valid for test not for tuning phase
        raise ArgparserException('`config_path` must be provided for tuning.')
