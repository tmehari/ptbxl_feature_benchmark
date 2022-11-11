import torch
import logging
import time
import os
import pdb 
from argparse import ArgumentParser

from clinical_ts.create_logger import create_logger
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap, eval_auc, eval_auc_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions


logger = create_logger(__name__)


def get_loss_from_output(out, key="loss"):
    # pdb.set_trace()
    return out[key] if isinstance(out, dict) else get_loss_from_output(out[0], key)


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack(
        [x[key1] for x in res if type(x) == dict and key1 in x.keys() and x[key1].numel() > 0]
    ).mean()


def cat(res, key):
    return torch.cat(
        [x[key] for x in res if type(x) == dict and key in x.keys()]
    ).numpy()


def evaluate_macro(preds, targets, idmap, lbl_itos, verbose=False):
    if len(preds) == 0:
        return 0,0
    idmap = idmap
    # for val sanity check TODO find cleaner solution
    idmap = idmap[: preds.shape[0]]

    scores = eval_auc(targets, preds, classes=lbl_itos, parallel=True)

    preds_agg, targs_agg = aggregate_predictions(preds, targets, idmap)

    scores_agg = eval_auc(targs_agg, preds_agg,
                              classes=lbl_itos, parallel=True)
    
    macro = scores["label_AUC"]["macro"]
    macro_agg = scores_agg["label_AUC"]["macro"]

    return macro, macro_agg


def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--gpus", help="number of gpus to use; use cpu if gpu=0", type=int, default=1
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--datasets",
        dest="target_folders",
        nargs="+",
        help="used datasets for training",
    )
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out_dim", type=int, default=71)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--input_size", type=int, default=250)
    parser.add_argument("--label_threshold", type=int, default=100)
    parser.add_argument("--test_only", type=bool, default=False)
    return parser


def init_logger(debug=False, log_dir="./experiment_logs"):
    level = logging.INFO

    if debug:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "info.log"),
        level=level,
        format="%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ",
    )
    return logging.getLogger(__name__)


def get_experiment_name(args):
    experiment_name = str(time.asctime()) + "_" + \
        str(time.time_ns())[-3:] 
    if args.input_size != 250:
        experiment_name += "_inputsize=" + str(args.input_size)
    return experiment_name


def load_from_checkpoint(pl_model, checkpoint_path):
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data
