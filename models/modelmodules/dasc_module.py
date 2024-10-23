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
import json
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
        self.train_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)

        # for averaging f1 across batches
        self.train_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        # for saving prediction results
        self.test_preds = []
        self.test_targets = []
        self.test_aspects = []


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
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log.info(f"epoch: {self.current_epoch}, lr: {current_lr :.6f}, train/loss: {self.train_loss.compute().item() :.4f}")
        

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
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1)  # update best so far val f1
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True, sync_dist=True)
        log.info("")
        log.info(f"val/loss: {self.val_loss.compute().item() :.4f}")
        log.info(f"val/acc: {self.val_acc.compute().item() :.4f}")
        log.info(f"val/f1: {self.val_f1.compute().item() :.4f}")
        

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
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # update predictions
        self.test_preds.extend(preds.cpu().numpy().tolist())
        self.test_targets.extend(targets.cpu().numpy().tolist())
        self.test_aspects.extend(batch["aspect_ids"].cpu().numpy().tolist())

        return {"loss": loss, "preds": preds, "targets": targets, "aspect_ids": batch["aspect_ids"]}
    
    def test_epoch_end(self, outputs):
        log.info('')
        log.info(f"test/acc: {self.test_acc.compute().item() :.4f}")
        log.info(f"test/f1: {self.test_f1.compute().item() :.4f}")
        
        output_path = self.conf.output_dir + "/preds-" + str(self.conf.model.arch) + '-' + str(self.conf.data.name) + ".json"
        with open(output_path, "w") as fw:
            json.dump(
                {
                    "preds": self.test_preds,
                    "label_ids": self.test_targets,
                    "aspect_ids": self.test_aspects,
                }, fw)

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
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        )