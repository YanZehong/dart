import pdb
import sys
import os
from webbrowser import get

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
import pytorch_lightning as pl
from .registry import register_module
import torch
from models import build_model
from torchmetrics import MaxMetric, MeanMetric, F1Score, Accuracy, Precision
from collections import OrderedDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.logger import get_logger

log = get_logger(__name__)


@register_module("DASC")
class DASCModule(pl.LightningModule):

    def __init__(self, conf, **kwargs) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.conf = conf
        self.net = build_model(conf=conf)
        self.criterion = torch.nn.CrossEntropyLoss()
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task = "binary")
        self.val_acc = Accuracy(task = "binary")
        self.test_acc = Accuracy(task = "binary")

        # for averaging f1 across batches
        self.train_f1 = F1Score(task = "binary", average="macro")
        self.val_f1 = F1Score(task = "binary", average="macro")
        self.test_f1 = F1Score(task = "binary", average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    # def on_train_epoch_start(self):
    #     pl.seed_everything(self.conf.seed + self.current_epoch)

    def training_step(self, batch, batch_idx):
        output = self.net(**batch)
        logits, targets = output["logits"], batch["label_ids"]
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `training_step()`
        log.info(f"train/loss: {self.train_loss.compute().item()}")
        

    def validation_step(self, batch, batch_idx):
        output = self.net(**batch)
        logits, targets = output["logits"], batch["label_ids"]
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1)  # update best so far val f1
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)
        

    def test_step(self, batch, batch_idx):
        output = self.net(**batch)
        logits, targets = output["logits"], batch["label_ids"]
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        train_conf = self.conf.train

        optimizer_list = []
        optimizer = AdamW(self.net.parameters(), lr=train_conf.lr)
        optimizer_list += [optimizer]

        scheduler_list = []
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=train_conf.scheduler_factor,
            patience=train_conf.scheduler_patience,
        )
        

        return (
            optimizer_list,
            {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        )