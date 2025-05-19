#import ml_collections
import torch
import torch.optim as optim
import pytorch_lightning as pl
import json
import os
import torch.distributed as dist
import warnings
import datetime

from data import MNISTSet, ShapeNet15kPointClouds, collate_fn
from torch.utils.data import DataLoader
from config import create_config
from model import SetFlowModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy

# torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42, workers=True)

config = create_config()
# timestamp = "shapenet"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
timestamp = f"mnist-{timestamp}"

@rank_zero_only
def setup_experiment_dir():
    os.makedirs(f"weights/{timestamp}", exist_ok=True)
    with open(f"weights/{timestamp}/config.json", "w") as f:
        f.write(config.to_json())

setup_experiment_dir()

checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,
    every_n_epochs=1,
    monitor="train_loss",
    dirpath=f"weights/{timestamp}",
    filename="checkpoint-{epoch:02d}-{train_loss:02f}",
)

model = SetFlowModule(config=config)
dataset = MNISTSet()
dataloader = DataLoader(dataset, batch_size=64, num_workers=32)
# dataset = ShapeNet15kPointClouds(categories=['airplane'], split='train')
# dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn, num_workers=32)

trainer = pl.Trainer(
    accelerator="gpu",
    # precision="bf16",
    # precision="64-true",
    devices=10,
    max_epochs=config.training.epochs,
    # gradient_clip_val=1.0,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
    # strategy=DDPStrategy(find_unused_parameters=True)
)

trainer.fit(
    model=model, 
    train_dataloaders=dataloader,
)
