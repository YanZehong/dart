import os
import hydra
import torch
from models import build_module
from datamodules import build_datamodule
from utils.hydra import replace_reference, fix_relative_path
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint, EarlyStopping, LearningRateMonitor

@hydra.main(config_path="./configs", config_name="cfg", version_base="1.12")
def main(conf):
    replace_reference(conf)
    fix_relative_path(conf)
    print(conf)
    print(f"################{_get_rank()} | {os.getcwd()} | {conf.output_dir}")

    module = build_module(conf=conf)
    datamodule = build_datamodule(conf=conf)
    pl.seed_everything(conf.seed)

    if conf.debug:
        print("################Debug Mode################")
        conf.train.num_workers = 0
        conf.train.prefetch_factor = 2
    
    if conf.test.monitor == "val/loss":
        decision_mode = "min"
    else:
        decision_mode = "max"

    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        ModelCheckpoint(dirpath=conf.output_dir,
                        monitor=conf.test.monitor,
                        mode=decision_mode),
        EarlyStopping(patience=conf.train.patience,
                      monitor=conf.test.monitor,
                      mode=decision_mode,
                      check_on_train_epoch_end=False),
    ]

    conf.gpu = [conf.gpu] if isinstance(conf.gpu, int) else conf.gpu
    trainer = pl.Trainer(
        max_epochs=conf.train.num_epoch,
        callbacks=callbacks,
        gradient_clip_val=conf.train.clip_grad,
        accelerator="gpu",
        devices=conf.gpu,
        overfit_batches=conf.overfit_batches,
        val_check_interval=conf.val_check_interval,
        num_sanity_val_steps=conf.num_sanity_val_steps,
        deterministic=conf.deterministic,
        strategy="ddp" if len(conf.gpu) > 1 else None,
    )
    
    trainer.fit(model=module, datamodule=datamodule)

    if len(conf.gpu) > 1:
        best_ckpt_path = callbacks[2].best_model_path
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = pl.Trainer(accelerator="gpu", devices=conf.gpu[0])
            # model = MyModel.load_from_checkpoint(best_ckpt_path)
            trainer.test(model=module, datamodule=datamodule, ckpt_path=best_ckpt_path)
    else:
        trainer.test(model=module, datamodule=datamodule, ckpt_path="best")

if __name__ == '__main__':
    main()