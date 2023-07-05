from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import ext_parser
import lit_model
import wandb
from lit_load_data import ImgDataModule


def objective(trial):  # current time to create log dir
    parser = ArgumentParser()
    parser = ext_parser.add_main_args(parser)
    # tmp_args, _ = parser.parse_known_args()
    cfg = parser.parse_args()

    loggers = []
    swing_neigh = 'swing_neigh' if cfg.swing_neigh else 'no_swing_neigh'
    wandb_logger = WandbLogger(name=f"{swing_neigh} test: {cfg.test_ids_}", project="parkinson", entity="jm8582", config=cfg)
    wandb.init()
    wandb.run.log_code(".")
    cfg = wandb.config
    cfg.update({"run_id": wandb.run.id}, allow_val_change=True)
    loggers.append(wandb_logger)

    cfg.targets = cfg.targets_str.split("_")

    Net = getattr(lit_model, cfg.lit_model)
    # parser = Net.add_model_specific_args(parser)
    before_intervals1 = torch.arange(0, cfg.before_interval_diff1 * cfg.before_interval_num1, cfg.before_interval_diff1)
    before_intervals2 = torch.arange(0, cfg.before_interval_diff2 * cfg.before_interval_num2, cfg.before_interval_diff2)
    before_intervals = torch.sort(torch.unique(torch.concat((before_intervals1, before_intervals2))))[0]
    cfg.in_dim = len(before_intervals)
    cfg.in_time_len = before_intervals[-1]
    before_intervals = torch.diff(before_intervals, prepend=torch.tensor([0])).long()

    cfg.fog_dim = len(np.unique(cfg.fog_map))
    cfg.act_dim = len(np.unique(cfg.act_map))

    # test id varies when using wandb sweep
    cfg.update({"test_ids": [cfg.test_ids_]}, allow_val_change=True)
    cfg.update({"train_ids": tuple({1, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16} - {cfg.test_ids_})}, allow_val_change=True)
    print(f"train id: {cfg.train_ids}")

    net = Net(**vars(cfg))
    net.cfg = cfg
    net.init_net()

    dset = ImgDataModule(cfg, before_intervals)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_sample_act_acc" if "act" in cfg.targets else "val_sample_fog_acc",
        mode="max",
        dirpath=f"./logs/{cfg.run_id}",
        filename="{epoch:02d}-{val_acc:.2f}",
        save_last=True,
    )

    trainer = pl.Trainer(
        default_root_dir=f"./logs/{cfg.run_id}",
        accelerator="gpu",
        devices=[0],
        max_epochs=cfg.max_epoch,
        logger=loggers,
        precision=cfg.precision,
        callbacks=[
            # EarlyStopping(monitor="val_acc_act", patience=20, mode="max"),
            checkpoint_callback,
            lr_monitor,
        ],
        fast_dev_run=False,
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        enable_checkpointing=True,
    )

    if cfg.mode == "train":
        trainer.fit(model=net, datamodule=dset, ckpt_path=cfg.model_ckpt)
    elif cfg.mode == "test":
        trainer.test(model=net, datamodule=dset, ckpt_path=cfg.model_ckpt)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    objective(None)
