from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from model import SAMRoadplus
import wandb
import lightning.pytorch as pl
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 사용을 위해 주석 처리

parser = ArgumentParser()
parser.add_argument(
    "--config",
    default="",
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default="", help="checkpoint of the last epoch of the model "
)
parser.add_argument("--precision", default=16, help="32 or 16")
parser.add_argument("--fast_dev_run", default=False, action="store_true")
parser.add_argument("--dev_run", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run
    load_dotenv()
    # start a new wandb run to track this script
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        # set the wandb project where this run will be logged
        entity=config.WANDB_TEAM_NAME,
        project=config.WANDB_PROJECT_NAME,
        name=config.WANDB_EXPERIMENT_NAME,
        # track hyperparameters and run metadata
        config={
            k: v
            for k, v in config.items()
            if k
            not in ["WANDB_TEAM_NAME", "WANDB_PROJECT_NAME", "WANDB_EXPERIMENT_NAME"]
        },
        # disable wandb if debugging
        mode="disabled" if dev_run else None,
    )
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadplus(config)

    train_ds, val_ds = SatMapDataset(
        config, is_train=True, dev_run=dev_run
    ), SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1, save_top_k=3, monitor="val_loss", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
        gradient_clip_val=1.0,
    )
    # trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
