import hydra
import torch
from models import build_module
from datamodules import build_datamodule
from utils.hydra import replace_reference, fix_relative_path
import pytorch_lightning as pl

@hydra.main(config_path="./configs", config_name="cfg", version_base="1.12")
def main(conf):
    replace_reference(conf)
    fix_relative_path(conf)
    print(conf)

    print("Instantiating model and datamodule")
    module = build_module(conf=conf)
    datamodule = build_datamodule(conf=conf)

    conf.gpu = [conf.gpu] if isinstance(conf.gpu, int) else conf.gpu
    trainer = pl.Trainer(
        accelerator="cpu" if not conf.gpu else "gpu",
        devices=1 if not conf.gpu else conf.gpu,
    )

    print("Starting testing!")
    trainer.test(model=module, datamodule=datamodule, ckpt_path=conf.ckpt_path)


if __name__ == "__main__":
    main()