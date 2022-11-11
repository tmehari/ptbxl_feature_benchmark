import time
import pickle
import yaml
import logging
import os
import sys
import re
import pdb
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW, SGD, Adam
from argparse import ArgumentParser
from torch import tensor 

from dl_models.ecg_resnet import ECGResNet


from clinical_ts.create_logger import create_logger
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap, eval_auc, eval_auc_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions


from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import get_loss_from_output, mean, cat, evaluate_macro, get_experiment_name, load_from_checkpoint, init_logger, parse_args

from pytorch_lightning import Trainer
from clinical_ts.ecg_datamodule import ECGDataModule
from os.path import join, dirname, exists
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from clinical_ts.data_modifiers import get_data_modifiers
logger = create_logger(__name__)


class BasicProgressBar(ProgressBarBase):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs):
        super().on_train_batch_end(trainer, pl_module, outputs)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')
        sys.stout.write("loss: {:.4f}".format(pl_module.current_loss))
        sys.stout.write("val_auc: {:.4f}".format(pl_module.current_val_auc))


class PL_Model(pl.LightningModule):
    def __init__(
        self,
        create_model,
        batch_size,
        num_samples,
        lr=0.001,
        opt_weight_decay=1e-6,
        loss_fn=F.binary_cross_entropy_with_logits,
        opt=AdamW,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
        """

        super(PL_Model, self).__init__()
        self.save_hyperparameters()
        self.model = self.init_model(create_model)
        self.epoch = 0
        self.current_loss = -1
        self.current_val_auc = None

        # save all final auc values for lbl_itos labels in val- and test scores
        self.val_scores = None
        self.test_scores = None
        self.save_results = False

    def init_model(self, create_model):
        model = create_model()
        return model

    def configure_optimizers(self):
        # optimizer = LARSWrapper(Adam(parameters, lr=self.hparams.lr))
        optimizer = self.hparams.opt(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.opt_weight_decay)
        return optimizer

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.hparams.loss_fn(preds, targets)
        self.current_loss = loss.item()
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = torch.tensor(
            [get_loss_from_output(output) for output in training_step_outputs]
        ).mean()
        self.log("train/total_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log("lr", self.hparams.lr, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        preds = torch.sigmoid(self(x))
        loss = self.hparams.loss_fn(preds, targets)
        results = {
            "val_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def test_step(self, batch, batch_idx):
        x, targets = batch
        preds = self(x)
        loss = self.hparams.loss_fn(preds, targets)
        results = {
            "test_loss": loss,
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def test_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        preds = torch.sigmoid(tensor(preds)).numpy()
        macro, macro_agg = evaluate_macro(
            preds, targets, self.trainer.datamodule.test_idmap, self.trainer.datamodule.lbl_itos, verbose=True)
        test_loss = mean(outputs, "test_loss")

        log = {
            "test/total_loss": test_loss,
            "test/test_macro": macro,
            "test/test_macro_agg": macro_agg,
            # 'test/scores_agg':scores_agg,
        }

        self.log_dict(log)
        return {"test_loss": test_loss, "log": log, "progress_bar": log}

    def validation_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        preds = torch.sigmoid(tensor(preds)).numpy()
        macro, macro_agg = evaluate_macro(
            preds, targets, self.trainer.datamodule.val_idmap, self.trainer.datamodule.lbl_itos, verbose=True)
        # pdb.set_trace()
        val_loss = mean(outputs, "val_loss")
        self.current_val_auc = macro_agg
        log = {
            "val/total_loss": val_loss,
            "val/val_macro": macro,
            "val/val_macro_agg": macro_agg,
            # 'val/scores_agg':scores_agg,
        }
        self.log_dict(log)

        return {"val_loss": val_loss, "log": log, "progress_bar": log}

    def on_train_start(self):
        self.epoch = 0

    def on_epoch_end(self):
        self.epoch += 1


def get_pl_model(args, datamodule):

    def create_model():
        return ECGResNet('xresnet1d50', datamodule.num_classes)

    model_fun = create_model

    pl_model = PL_Model(
            model_fun,
            args.batch_size,
            datamodule.num_samples
        )
    return pl_model


def cli_main():
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    
    lbl_itos_arr = pickle.load(open(join(args.target_folders[0] ,"lbl_itos.pkl"), "rb"))

    #label_class, lbl_itos = lbl_itos_arr[args.label_class]
    lbl_itos = lbl_itos_arr[args.label_class]
    label_class = args.label_class
    experiment_name = get_experiment_name(args)

    
    init_logger(log_dir=join(args.logdir, experiment_name))
    logger.info(dict(args.__dict__))
    data_modifiers = get_data_modifiers(args, label_class, lbl_itos)

    datamodule = ECGDataModule(
        args.batch_size,
        args.target_folders,
        data_modifiers,
        lbl_itos=lbl_itos,
        data_input_size=args.input_size,
    )

    # configure trainer
    tb_logger = TensorBoardLogger(
        args.logdir, name=experiment_name, version="",) if not args.test_only else None

    trainer = Trainer(
        logger=tb_logger,
        max_epochs=args.epochs,
        gpus=args.gpus,
        callbacks=[ModelCheckpoint(monitor='val/total_loss')]
        # callbacks=[BasicProgressBar()]
    )

    pl_model = get_pl_model(args, datamodule)

    # load checkpoint
    if args.checkpoint_path != "":
        if exists(args.checkpoint_path):
            logger.info("Retrieve checkpoint from " + args.checkpoint_path)
            # pl_model.load_from_checkpoint(args.checkpoint_path)
            load_from_checkpoint(pl_model, args.checkpoint_path)
        else:
            raise ("checkpoint does not exist")

    # start training
    if not args.test_only:
        trainer.fit(pl_model, datamodule)
        trainer.save_checkpoint(os.path.join(
            args.logdir, experiment_name, "checkpoints", "model.ckpt"))

    
    _ = trainer.validate(pl_model, datamodule=datamodule)
    _ = trainer.test(pl_model, datamodule=datamodule)

    val_scores = pl_model.val_scores
    test_scores = pl_model.test_scores
    filename = "score.pkl"
    scores_file = join(dirname(args.checkpoint_path), filename) if args.checkpoint_path != "" else join(
        args.logdir, experiment_name, "checkpoints", filename)
    
    prefix = '_label=ptb' if 'ptb' in args.label_class else ''
    prefix = '_label=12sl' if '12sl' in args.label_class else prefix
    
    scores = pickle.load(open(scores_file, "rb")
                         ) if exists(scores_file) else {}
    if not args.test_only:
        scores['description'] = "training=" + prefix 
    scores[prefix] = {"val_scores": val_scores, "test_scores": test_scores}
    pickle.dump(scores, open(scores_file, "wb"))


if __name__ == "__main__":
    cli_main()
