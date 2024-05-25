import argparse
import os
from pathlib import Path

import torch
from dataclasses import dataclass, asdict
# from lightning.pytorch import loggers as pl_loggers
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from data.camcan_datamodule import CamCANDataModule, CamCANDataModule2D
from data.nlst_datamodule import NLSTDataModule
from models.lightning_module import Registration
from utils.utils import read_config, create_ckpt_directory
from utils.params import Params, process_params
import wandb
import yaml
from utils.path_utils import get_data_paths
import shutil


DEFAULT_CONFIG_FILE_NAME = "config3d.yaml"


# TODO: move this to utils after refactoring, circular reports need to be resolved
def get_datamodule(params):

    if params.dataset == 'brain_camcan':
        data_module = CamCANDataModule(**params.__dict__, crop_size=[176, 192, 176])
    elif params.dataset == 'brain_camcan2d':
        data_module = CamCANDataModule2D(**params.__dict__, crop_size=[192, 192])
    elif params.dataset == 'nlst':
        data_module = NLSTDataModule(**params.__dict__, crop_size=[96, 96, 96])
    else:
        raise ValueError(f"Unknown dataset: {params.dataset}")

    return data_module


def parser_command_line():
    "Define the arguments required for the script"
    parser = argparse.ArgumentParser(description="Attention-Conditioned Neural Implicit Segmentation Representation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest='pipeline', help='pipeline to run')
    # Arguments for training
    parser_train = subparser.add_parser('train', help='train the model')
    parser_train.add_argument('-c', '--config', default=f'configs/{DEFAULT_CONFIG_FILE_NAME}',
                              help='config file (.yaml) containing the custom hyper-parameters.')
    parser_train.add_argument('-s', '--save_path', default=None,
                              help='Directory where to save model directory (to save ckpts, config, results, etc)')

    # Arguments for evaluation
    parser_eval = subparser.add_parser('eval', help='evaluate the model')
    parser_eval.add_argument('-cp', '--ckpt', help='Path to checkpoint', default=None)
    parser_eval.add_argument('-c', '--config', default=None,
                             help='Custom config file (.yaml) containing the hyper-parameters for inference. '
                                  'By default we try to load config file from provided checkpoint directory.')
    parser_eval.add_argument('-s', '--save_path', default=None,
                             help='Directory where to save model directory (to save ckpts, config, results, etc)')
    return parser.parse_args()


def main():
    args = parser_command_line()
    assert args.pipeline in {"train", "eval"}
    if args.pipeline == "train":
        cfg = {}
        if args.config is not None:
            cfg = read_config(args.config)
        params = Params(**cfg)
    else:
        cfg = {}
        try:
            cfg_path = Path(args.ckpt).parent.parent / DEFAULT_CONFIG_FILE_NAME
            cfg = read_config(cfg_path)
            print("Config successfully found in checkpoint directory.")
        except FileNotFoundError:
            print("Config file not found in checkpoint directory. Loading default params.")
        if args.config is not None:
            print("You have passed a custom config, we will overwrite the provided fields.")
            cfg_custom = read_config(args.config)
            cfg = {**cfg, **cfg_custom}  # Overwrite variables in original cfg
        params = Params(**cfg)
        params.batch_size = 1
        params.num_workers = 1
    process_params(params)

    # create checkpoint directory
    if args.save_path is None:
        model_dir = Path().cwd() / "ckpts"
    else:
        model_dir = args.save_path
    model_dir = model_dir / args.pipeline
    current_dir = create_ckpt_directory(str(model_dir))
    params.current_dir = current_dir

    # Dump current params to yaml in model save directory
    config_file = f'{current_dir}/config.yaml'
    with open(config_file, 'w') as file:
        yaml.dump(params.__dict__, file)

    # comment in for debugging purposes
    os.environ['WANDB_DISABLED'] = params.wandb_disable

    if params.is_sweep:
        # TODO is this still supported?
        # load sweep config and assign the values
        with open("./sweep.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        wandb.init(project='GeoReg', entity=ENTITY, config=config)
        params.lr = wandb.config.lr
    else:
        wandb.init(project='GeoReg', entity=ENTITY, config=asdict(params))

    # use only one GPU
    if isinstance(params.gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    # Print available GPU VRAM
    print("Cuda available:", torch.cuda.is_available(), "  Available VRAM:", torch.cuda.mem_get_info()[1]/1024**3)

    params.data_dir = get_data_paths(params.dataset)
    print(params.data_dir)

    data_module = get_datamodule(params)
    data_module.setup('fit')

    params.spatial_dims = data_module.train_dataset[0]["moving"].shape
    params.num_seg_classes = data_module.train_dataset.num_seg_classes

    # configure logger
    # logger = pl_loggers.WandbLogger(save_dir=model_dir)

    # model checkpoint callback with ckpt metric logging
    callbacks = [ModelSummary(max_depth=4)]
    if args.pipeline == 'train':
        # ckpt_callback = ModelCheckpoint(save_last=True,
        #                                 dirpath=f'{current_dir}/checkpoints',
        #                                 verbose=True
        #                                 )
        ckpt_callback = ModelCheckpoint(save_top_k=3,
                                        save_last=True,
                                        dirpath=f'{current_dir}/checkpoints',
                                        verbose=True,
                                        monitor='val_metrics/mean_dice'
                                        )
        callbacks.append(ckpt_callback)

    trainer = Trainer(default_root_dir=current_dir,
                      # logger=logger,
                      accumulate_grad_batches=4,
                      callbacks=callbacks,
                      accelerator='gpu',
                      devices=1,
                      max_epochs=params.max_epochs,
                      check_val_every_n_epoch=params.check_val_every_n_epoch,
                      )

    if args.pipeline == 'train':
        # lightning model
        model = Registration(hparams=params.__dict__)
        # run training
        trainer.fit(model, datamodule=data_module)
    elif args.pipeline == 'eval':
        model = Registration.load_from_checkpoint(args.ckpt)
        current_dir = Path(args.ckpt).parent.parent
        model.hparams.current_dir = str(current_dir)
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
