import os
import torchvision
from torchvision.transforms import transforms
import torch
from dataclasses import dataclass, asdict
from models.models import RegistrationTransformer
from utils.visualization import visualise_results
from lightning.pytorch import loggers as pl_loggers
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from models.lightning_module import Registration
from data.mnist_datamodule import MNISTDataModule
from data.camcan_datamodule import CamCANDataModule
import wandb
import yaml
from typing import List


@dataclass
class Params:
    # Dataloader
    num_train: int = 100
    num_val: int = 10
    num_test: int = 10
    batch_size: int = 8
    num_workers: int = 8
    # Network
    num_hidden_layers: int = 8
    enc_num_hidden_layers: int = 8
    deform_num_hidden_layers: int = 4
    hidden_size: int = 128
    channels: tuple = (32, 32)
    pool_size: int = 2
    num_neigh: int = 5
    #wandb
    wandb_disable: str = "False"
    is_sweep: bool = False
    # gpu
    gpu: int = 0
    # training
    lr: float = 1e-4
    # Trainer
    check_val_every_n_epoch: int = 1
    max_epochs: int = 100
    # regularisation
    alpha: float = 0.01
    reg_type: str = 'l2'
    #loss
    loss_type: str = 'ncc'


def main():
    params = Params()
    model_dir = os.getcwd()

    if params.is_sweep:
        # load sweep config and assign the values
        with open("./sweep.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        wandb.init(project='NewSota', config=config)
        params.lr = wandb.config.lr
    else:
        wandb.init(project='NewSota', config=asdict(params))

    # comment in for debugging purposes
    os.environ['WANDB_DISABLED'] = params.wandb_disable

    # use only one GPU
    gpus = None if params.gpu is None else 1
    if isinstance(params.gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    data_module = CamCANDataModule(batch_size=params.batch_size)
    data_module.setup('fit')

    # lightning model
    model = Registration(hparams=params.__dict__, channels=params.channels, pool_size=params.pool_size, num_neigh=params.num_neigh)
    # configure logger
    logger = pl_loggers.WandbLogger(save_dir=model_dir)

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = ModelCheckpoint(save_last=True,
                                      dirpath=f'{model_dir}/checkpoints/',
                                      verbose=True
                                    )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      callbacks=[ckpt_callback],
                      accelerator='gpu',
                      devices=1,
                      max_epochs=params.max_epochs,
                      check_val_every_n_epoch=params.check_val_every_n_epoch,
                      )

    # run training
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
