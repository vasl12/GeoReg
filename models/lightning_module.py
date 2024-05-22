from lightning import LightningModule
from models.models import RegistrationTransformer
import torch
from torch import nn
from torch.optim import Adam
from transformation.transformation import warp
from utils.visualization import visualise_results
from utils.utils import measure_seg_metrics
from losses.regularization import l2reg_loss, bending_energy_loss
from losses.loss import NCC
import wandb


class Registration(LightningModule):
    def __init__(self, hparams, channels=(16,), pool_size=1, num_neigh=5):
        super(Registration, self).__init__()
        self.hparams.update(hparams)

        # in channels: 1 for grayscale
        # channels of the unet layers. Length of the tuple defines the amount of layers
        self.network = RegistrationTransformer(in_channels=1, channels=channels, pool_size=pool_size, num_neigh=num_neigh)
        # TODO: fix the paths and move thhis in a datamodule
        if self.hparams.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.hparams.loss_type == 'ncc':
            self.loss_fn = NCC()
        if self.hparams.reg_type == 'bending':
            self.reg = bending_energy_loss
        elif self.hparams.reg_type == 'l2':
            self.reg = l2reg_loss

        self.validation_step_output = []

    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, tar, src):
        disp = self.network(tar, src)
        return disp

    def _step(self, batch, mode='train'):
        """ Forward pass inference + compute loss """

        fixed = batch['fixed']
        moving = batch['moving']
        out = self.forward(fixed, moving)

        disp = out[-1].permute(0, 3, 1, 2)
        # only output disp field
        batch['warped_moving'] = warp(moving, disp)
        warped_moving_seg = warp(batch['moving_seg'], disp, interp_mode='nearest')

        reg_loss = 0
        for disp in out:
            disp = disp.permute(0, 3, 1, 2)
            reg_loss += self.hparams.alpha * self.reg(disp)

        intensity_loss = self.loss_fn(fixed, batch['warped_moving'])
        losses = intensity_loss + reg_loss

        # visualisation in wandb
        wandb.log({f"{mode}/img_loss": intensity_loss.clone().detach().cpu().numpy().item()})
        wandb.log({f"{mode}/reg_loss": reg_loss.clone().detach().cpu().numpy().item()})

        results = measure_seg_metrics(batch['fixed_seg'].clone().detach(), warped_moving_seg.clone().detach())
        wandb.log({f'metrics/{k}': metric.item() for k, metric in results.items()})

        # visualise only the first batch
        fixed = batch['fixed'][0].clone().detach().squeeze(0).cpu()
        moving = batch['moving'][0].clone().detach().squeeze(0).cpu()
        fixed_seg = batch['fixed_seg'][0].clone().detach().squeeze(0).cpu()
        moving_seg = batch['moving_seg'][0].clone().detach().squeeze(0).cpu()
        warped_moving_seg = warped_moving_seg[0].clone().detach().squeeze(0).cpu()
        warped_moving = batch['warped_moving'][0].clone().detach().squeeze(0).cpu()
        field = out[-1].permute(0, 3, 1, 2)[0].clone().detach().cpu()

        fig1, fig2 = visualise_results(fixed, moving, warped_moving, field, show=False, cmap='viridis')
        self.logger.experiment.log({f'{mode} intensities': fig1,
                                           "global_step": self.global_step})

        self.logger.experiment.log({f'{mode} def fields': wandb.Image(fig2),
                                           "global_step": self.global_step})

        fig3, _ = visualise_results(fixed_seg, moving_seg, warped_moving_seg, field, show=False, cmap='viridis')
        self.logger.experiment.log({f'{mode} segmentations': fig3,
                                    "global_step": self.global_step})

        step_outputs = {'disp_pred': disp[-1], 'warped_moving': batch['warped_moving']}
        return losses, step_outputs

    def training_step(self, batch, batch_idx):
        train_loss, _ = self._step(batch, mode='train')
        return train_loss

    # TODO: add validation step
    def validation_step(self, batch, batch_idx):
        val_loss, step_outputs = self._step(batch, mode='val')
        self.validation_step_output.append(val_loss)
