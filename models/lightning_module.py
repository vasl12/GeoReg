import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
from kornia.filters.sobel import spatial_gradient, spatial_gradient3d
from lightning import LightningModule
from models.models import RegistrationTransformer
import torch
from torch import nn
from torch.optim import Adam
from transformation.transformation import warp, resample
from utils.visualization import visualise_results_2d, visualise_features, visualise_displ_components
from utils.metric import measure_metrics, measure_seg_metrics, MetricReporter, calculate_dice, \
    calculate_dice_no_background
from losses.loss import NCC, CosineSimLoss, LNCCLoss, L4, NMI, PerceptualLoss
# from monai.losses import PerceptualLoss

import wandb
# from torchmetrics.image import StructuralSimilarityIndexMeasure


MODE_LUT = {"val": "Validation",
            "train": "Training",
            "test": "Testing"}


class Registration(LightningModule):
    def __init__(self, hparams):
        super(Registration, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)

        self.train_log_rate = self.hparams.check_val_every_n_epoch

        # in channels: 1 for grayscale
        self.num_spatial_dims = len(self.hparams.spatial_dims) - 1
        self.network = RegistrationTransformer(in_channels=self.hparams.spatial_dims[0],
                                               num_spatial_dims=self.num_spatial_dims, **hparams)

        self.loss_fn = self._get_loss_function(self.hparams.loss_function, window_size=self.hparams.window_size,
                                               spatial_dims=self.num_spatial_dims)
        # Loss weights per resolution
        self.do_order0_reg = any([i > 0.0 for i in self.hparams.reg_order0_alpha])
        self.do_order2_reg = any([i > 0.0 for i in self.hparams.reg_order2_alpha])
        self.do_consist_reg = any([i > 0.0 for i in self.hparams.reg_consist_alpha])

        self.pool_func = nn.functional.avg_pool2d if self.num_spatial_dims == 2 else nn.functional.avg_pool3d
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.metric_reporter = None

    def on_fit_start(self):
        self.metric_reporter = MetricReporter(id_list=os.listdir(f'{Path(self.hparams.data_dir) / "test"}'), save_dir=f'{Path(self.hparams.current_dir) / "analysis"}')
        os.makedirs(f'{Path(self.hparams.current_dir) / "analysis"}', exist_ok=True)
        self.logger.log_hyperparams(self.hparams)

    def on_test_start(self) -> None:
        self.metric_reporter = MetricReporter(id_list=os.listdir(f'{Path(self.hparams.data_dir) / "test"}'), save_dir=f'{Path(self.hparams.current_dir) / "analysis"}')
        os.makedirs(f'{Path(self.hparams.current_dir) / "analysis"}', exist_ok=True)

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def _get_loss_function(loss_function, window_size=9, spatial_dims=3):
        if loss_function == 'mse':
            loss_fn = nn.MSELoss()
        elif loss_function == 'ncc':
            loss_fn = NCC()
        elif loss_function == 'l4':
            loss_fn = L4()
        elif loss_function == 'lncc':
            loss_fn = LNCCLoss(window_size=window_size)
        elif loss_function == 'cos':
            loss_fn = CosineSimLoss(dim=1)
        elif loss_function == 'nmi':
            loss_fn = NMI((0.0, 1.0))  # TODO: Not sure if using this correctly
        elif loss_function == 'perceptual':
            is_fake_3d = False
            loss_fn = PerceptualLoss()
            # loss_fn = PerceptualLoss(spatial_dims=spatial_dims, network_type='medicalnet_resnet50_23datasets', is_fake_3d=is_fake_3d)
        else:
            raise ValueError(f'Similarity loss config ({loss_function}) not recognised.')
        return loss_fn

    @staticmethod
    def _get_coords_per_res(coord_per_res_components: Dict[str, List[torch.Tensor]], fine_to_coarse=True) \
            -> Tuple[List[torch.Tensor], ...]:
        key_order = ("start", "inh", "par", "cross", "sib", "final")
        coords = [coord_per_res_components[k] for k in key_order]
        if fine_to_coarse:
            coords = [i[::-1] for i in coords]
        return tuple(coords)

    def _regularize_comp_order0(self, coord_per_res_components):
        (start_c_per_res, inh_c_per_res, par_c_per_res,
         cross_c_per_res, sib_c_per_res, final_c_per_res) \
            = self._get_coords_per_res(coord_per_res_components, fine_to_coarse=True)
        reg_loss = torch.zeros((1,), dtype=torch.float32, device=start_c_per_res[0].device)
        if not self.do_order0_reg:
            return reg_loss

        alphas = self.hparams.reg_order0_alpha
        pool_sizes = [1, *self.network.encoder.pool_sizes]
        res_weight = 1.0
        # ------------------ PARENT REGULARIZATION ------------------------------
        # Regularize the displacement from: coord they inherited -> where parent told this child to go
        for pool_size, alpha, c_start, c_par in zip(pool_sizes, alphas, start_c_per_res, par_c_per_res):
            # c_start = c_start.detach()  # We use .detach() to constrain reg to this layer
            disp = c_par - c_start
            reg_loss += (disp * disp).sum(-1).mean() * alpha * res_weight

        # ------------------ CROSS & SIBLING REGULARIZATION ------------------------------
        res_weight = 1.0
        # Regularize the displacement for each individual deformation in the par->cross->sib->...->cross->sib chain
        for pool_size, alpha, c_start, c_cross_list, c_sib_list in zip(pool_sizes, alphas, par_c_per_res,
                                                                       cross_c_per_res, sib_c_per_res):
            res_weight /= pool_size
            for j, (c_cross, c_sib) in enumerate(zip(c_cross_list, c_sib_list)):
                # c_start = c_start.detach()  # We use .detach() to constrain reg to this layer
                disp = c_cross - c_start
                reg_loss += (disp * disp).sum(-1).mean() * alpha * res_weight
                # c_cross = c_cross.detach()
                disp = c_sib - c_cross
                reg_loss += (disp * disp).sum(-1).mean() * alpha * res_weight
                c_start = c_sib
        return reg_loss

    def _regularize_comp_order2(self, coord_components):
        start_c_per_res, inh_c_per_res, par_c_per_res, cross_c_per_res, sib_c_per_res, final_c_per_res \
            = self._get_coords_per_res(coord_components, fine_to_coarse=True)
        loss = torch.zeros((1,), dtype=torch.float32, device=start_c_per_res[0].device)
        if not self.do_order2_reg:
            return loss
        alphas = self.hparams.reg_order2_alpha
        pool_sizes = [1, *self.network.encoder.pool_sizes]
        # res_weight = 1.0
        # # ------------------ PARENT REGULARIZATION ------------------------------
        # # Regularize the displacement from: coord they inherited -> where parent told this child to go
        # for pool_size, alpha, c_start, c_par in zip(pool_sizes, alphas, start_c_per_res, par_c_per_res):
        #     res_weight /= pool_size
        #     # c_start = c_start.detach()  # We use .detach() to constrain reg to this layer
        #     disp = c_par - c_start
        #     if len(disp.shape[1:-1]) == 2:
        #         filtered = spatial_gradient(disp.moveaxis(-1, 1), mode="diff", order=2, normalized=False)
        #     else:
        #         filtered = spatial_gradient3d(disp.moveaxis(-1, 1), mode="diff", order=2)
        #     loss += (filtered * filtered).sum((1, 2)).mean() * alpha * res_weight

        # ------------------ CROSS & SIBLING REGULARIZATION ------------------------------
        start_cs = start_c_per_res if self.hparams.reg_order2_from_start else inh_c_per_res
        res_weight = 1.0
        # Regularize the displacement for each individual deformation in the par->cross->sib->...->cross->sib chain
        for pool_size, alpha, c_start, c_final_list in zip(pool_sizes, alphas, start_cs, sib_c_per_res):
            res_weight /= pool_size
            for c_final in c_final_list:
                if self.hparams.reg_order2_detach:
                    c_start = c_start.detach()  # We use .detach() to constrain reg to this layer
                disp = c_final - c_start
                if len(disp.shape[1:-1]) == 2:
                    filtered = spatial_gradient(disp.moveaxis(-1, 1), mode="diff", order=2, normalized=False)
                else:
                    filtered = spatial_gradient3d(disp.moveaxis(-1, 1), mode="diff", order=2)
                loss += (filtered * filtered).sum((1, 2)).mean() * alpha * res_weight
        return loss

    def _calculate_displ_consist_loss(self, coord_components):
        """ Compute the difference between displacement fields across subsequent resolution."""
        start_c_per_res, _, _, _, _, final_c_per_res \
            = self._get_coords_per_res(coord_components, fine_to_coarse=True)
        loss = torch.zeros((1,), dtype=torch.float32, device=start_c_per_res[0].device)
        if not self.do_consist_reg:
            return loss

        pool_sizes = self.network.encoder.pool_sizes
        alphas = self.hparams.reg_consist_alpha
        ref_d = final_c_per_res[0] - start_c_per_res[0]
        for alpha, pool_size, start_c, final_c in zip(alphas, pool_sizes, start_c_per_res[1:], final_c_per_res[1:]):
            ref_d = self.pool_func(ref_d.moveaxis(-1, 1), pool_size, ceil_mode=True).moveaxis(1, -1)
            loss += torch.nn.functional.mse_loss(final_c - start_c, ref_d) * alpha
            ref_d = final_c - start_c
        return loss

    def _calculate_pixel_loss(self, fixed, moving, coord_components):
        start_c_per_res, _, par_c_per_res, cross_c_per_res, sib_c_per_res, final_c_per_res \
            = self._get_coords_per_res(coord_components, fine_to_coarse=True)
        fi = fixed.clone()
        loss = torch.zeros((1,), dtype=fi.dtype, device=fi.device)

        pool_sizes = self.network.encoder.pool_sizes
        alphas = self.hparams.loss_weight
        for i, (alpha, par_c, cross_c, sib_c) \
                in enumerate(zip(alphas, par_c_per_res, cross_c_per_res, sib_c_per_res)):
            if alpha == 0:
                continue
            # resample for every resolution to compute the loss
            if i != 0:
                pool_size = pool_sizes[i - 1]
                fi = self.pool_func(fi, pool_size, ceil_mode=True)
            # loss += self.loss_fn(resample(moving, par_c, interp_mode="bilinear"), fi) * alpha
            for c, s in zip(cross_c, sib_c):
                # loss += self.loss_fn(resample(moving, c, interp_mode="bilinear"), fi) * alpha
                loss += self.loss_fn(resample(moving, s, interp_mode="bilinear"), fi) * alpha
        return loss
    
    def _calculate_feat_loss(self, fixed, moving, coord_components):
        _, _, par_c_per_res, cross_c_per_res, sib_c_per_res, final_c_per_res \
            = self._get_coords_per_res(coord_components, fine_to_coarse=True)
        fi = fixed.clone()
        loss = torch.zeros((1,), dtype=fi.dtype, device=fi.device)

        pool_sizes = self.network.encoder.pool_sizes
        alphas = self.hparams.loss_weight
        for i, (alpha, par_c, cross_c, sib_c) \
                in enumerate(zip(alphas, par_c_per_res, cross_c_per_res, sib_c_per_res)):
            if alpha == 0:
                continue
            if i != 0:
                pool_size = pool_sizes[i - 1]
                fi = self.pool_func(fi, pool_size, ceil_mode=True)
            # resample for every resolution to compute the loss
            loss += self.loss_fn(resample(moving, par_c, interp_mode="bilinear"), fi) * alpha
            for c, s in zip(cross_c, sib_c):
                loss += self.loss_fn(resample(moving, c, interp_mode="bilinear"), fi) * alpha
                loss += self.loss_fn(resample(moving, s, interp_mode="bilinear"), fi) * alpha
        return loss
    
    def _calculate_disp_loss(self, batch, coord_components):
        start_c_per_res, _, par_c_per_res, cross_c_per_res, sib_c_per_res, final_c_per_res \
            = self._get_coords_per_res(coord_components, fine_to_coarse=True)
        d_gt = batch["disp"].moveaxis(-1, 1)
        loss = torch.zeros((1,), dtype=d_gt.dtype, device=d_gt.device)

        pool_sizes = self.network.encoder.pool_sizes
        alphas = self.hparams.loss_weight
        for i, (alpha, start_c, par_c, cross_c, sib_c) \
                in zip(alphas, start_c_per_res, par_c_per_res, cross_c_per_res, sib_c_per_res):
            if alpha == 0:
                continue
            if i != 0:
                pool_size = pool_sizes[i - 1]
                d_gt = self.pool_func(d_gt, pool_size, ceil_mode=True)
            gt_coord = start_c + d_gt.moveaxis(1, -1)
            loss += self.loss_fn(par_c, gt_coord) * alpha
            for c, s in zip(cross_c, sib_c):
                loss += self.loss_fn(c, gt_coord) * alpha
                loss += self.loss_fn(s, gt_coord) * alpha
        return loss

    def forward(self, tar, src):
        return self.network(tar, src)

    def _step(self, batch, batch_idx, mode='train'):
        """ Forward pass inference + compute loss """

        # for k, x in batch.items():
        #     if not isinstance(x, torch.Tensor):
        #         batch[k] = x.as_tensor()

        fixed = batch['fixed']
        moving = batch['moving']
        coord_components, out_features = self.forward(moving, fixed)

        reg0_loss = self._regularize_comp_order0(coord_components)
        reg2_loss = self._regularize_comp_order2(coord_components)
        cons_loss = self._calculate_displ_consist_loss(coord_components)
        reg_loss = reg0_loss + reg2_loss + cons_loss

        if self.hparams.loss_type == "pixel":
            sup_loss = self._calculate_pixel_loss(fixed, moving, coord_components)
        elif self.hparams.loss_type == "feat":
            sup_loss = self._calculate_feat_loss(fixed, moving, coord_components, out_features)
        elif self.hparams.loss_type == "disp":
            sup_loss = self._calculate_disp_loss(batch, coord_components)
        else:
            raise ValueError(f"Loss type {self.hparams.loss_type} not recognised")

        losses = sup_loss + reg_loss

        wandb.log({f"{mode}_metrics/supervised_loss": sup_loss.detach().cpu().item()})
        wandb.log({f"{mode}_metrics/reg_loss": reg_loss.detach().cpu().item()})
        wandb.log({f"{mode}_metrics/cons_loss": cons_loss.detach().cpu().item()})
        wandb.log({f"{mode}_metrics/reg0_loss": reg0_loss.detach().cpu().item()})
        wandb.log({f"{mode}_metrics/reg2_loss": reg2_loss.detach().cpu().item()})
        wandb.log({f"{mode}_metrics/total_loss": losses.detach().cpu().item()})
        self._do_logging(batch, batch_idx, coord_components, out_features, mode)
        return losses

    @torch.no_grad()
    def _do_logging(self, batch, batch_idx, coord_components, out_features, mode):
        if mode == "train":
            return

        # visualisation in wandb
        final_disp = coord_components["final"][-1] - coord_components["start"][-1]
        final_coord = coord_components["final"][-1]
        batch['warped_moving'] = resample(batch['moving'], final_coord, interp_mode="bilinear")
        seg_class_max = None
        if 'moving_seg' in batch.keys():
            batch['warped_moving_seg'] = resample(batch['moving_seg'], final_coord, interp_mode="nearest")
            seg_class_max = self.hparams.num_seg_classes

        shape = torch.tensor(batch["fixed"].shape[2:])

        warped_moving_gt_disp = None
        warped_moving_seg_gt_disp = None
        warped_moving_gt_disp_affine = None
        warped_moving_seg_gt_disp_affine = None
        field_gt_disp = None
        # log gt metrics if supervision is on disp level
        if "disp" in batch:
            field_gt_disp = batch["disp"][0].cpu() / 2 * shape
            batch['warped_moving_disp'] = warp(batch['moving'], batch["disp"], interp_mode='bilinear', normalise=False)
            warped_moving_gt_disp = batch['warped_moving_disp'][0, 0].detach().cpu()

        if "disp_affine" in batch:
            batch['warped_moving_disp_affine'] = warp(batch['moving'], batch["disp_affine"],
                                                      interp_mode='bilinear', normalise=False)
            warped_moving_gt_disp_affine = batch['warped_moving_disp_affine'][0, 0].detach().cpu()
            if 'moving_seg' in batch:
                batch['warped_moving_seg_disp_affine'] = warp(batch['moving_seg'], batch["disp_affine"],
                                                              interp_mode='nearest')

        batch['disp_pred'] = final_disp.detach() / 2 * shape.to(final_disp.device)
        batch['disp_norm'] = final_disp.detach()

        # measure results
        results = measure_metrics(batch, self.hparams.evaluation_metrics, seg_class_max=seg_class_max)
        wandb.log({f'{mode}_metrics/{k}': metric.item() for k, metric in results.items()})
        self.log_dict({f'{mode}_metrics/mean_dice': results['mean_dice'].item()})


        if mode == "test":
            self.metric_reporter.collect({f'{mode}_metrics/{k}': metric.item() for k, metric in results.items()})

        if mode == "val" and batch_idx > 4:
            return

        # log the visualizations for the intensity images
        fixed = batch['fixed'][0].detach().squeeze(0).cpu()
        moving = batch['moving'][0].detach().squeeze(0).cpu()
        warped_moving = batch['warped_moving'][0].detach().squeeze(0).cpu()
        batch['disp_pred'] = batch['disp_pred'].cpu()
        field = batch['disp_pred'][0]
        batch['disp_norm'] = batch['disp_norm'].cpu()

        error_warp_fix = (warped_moving - fixed).pow(2).sum(1).mean().item()
        error_mov_fix = (moving - fixed).pow(2).sum(1).mean().item()
        error_warp_2_disp_affine = None
        error_disp_affine_2_fix = None
        if warped_moving_gt_disp is not None:
            error_warp_2_disp_affine = (warped_moving - warped_moving_gt_disp_affine).pow(2).sum(1).mean().item()
            error_disp_affine_2_fix = (warped_moving_gt_disp_affine - fixed).pow(2).sum(1).mean().item()

        self._log_visualizations(fixed, moving, warped_moving, field, warped_moving_gt_disp_affine, field_gt_disp,
                                 error_warp_fix, error_mov_fix, error_warp_2_disp_affine, error_disp_affine_2_fix,
                                 shape, vmax=1.0, mode=mode, modality='intensities', metric_name="Pixel L2")

        # log the segmentation metrics and the segmentation visualization
        if 'moving_seg' in batch.keys() and batch['moving_seg'] is not None and "seg_metrics" in self.hparams.evaluation_metrics:
            fixed_seg = batch['fixed_seg'][0].detach().squeeze(0)
            moving_seg = batch['moving_seg'][0].detach().squeeze(0)
            warped_moving_seg = batch["warped_moving_seg"][0].detach().squeeze(0)

            # log the visualizations
            dice_warp_2_fix = results['mean_dice'].item()
            dice_mov_2_fix = results['mean_dice_before'].item()
            dice_warp_2_disp_affine = None
            dice_disp_affine_2_fix = None

            if "disp_affine" in batch:
                warped_moving_seg_gt_disp_affine = batch['warped_moving_seg_disp_affine'][0, 0].detach()
                dice_warp_2_disp_affine = calculate_dice_no_background(warped_moving_seg, warped_moving_seg_gt_disp_affine)
                dice_disp_affine_2_fix = calculate_dice_no_background(warped_moving_seg_gt_disp_affine, fixed_seg)

            self._log_visualizations(fixed_seg, moving_seg, warped_moving_seg, field,
                                     warped_moving_seg_gt_disp_affine, field_gt_disp,
                                     dice_warp_2_fix, dice_mov_2_fix, dice_warp_2_disp_affine, dice_disp_affine_2_fix,
                                     shape, seg_class_max,
                                     mode=mode, modality='segmentations', metric_name="Dice", cmap='viridis')

        # log the multi-resolution layer-wise deformation
        self._log_multires_layerwise_def(batch, coord_components, mode=mode)

    def _log_visualizations(self, fixed, moving, warped_moving, field, warped_moving_gt, field_gt,
                            error_warp_fix, error_mov_fix, error_warp_warpgt, error_warpgt_fix,
                            shape, vmax, mode='train', modality='intensities', metric_name='Pixel L2', cmap="gray"):
        fixed, moving, warped_moving, field = fixed.detach().cpu(), moving.detach().cpu(), warped_moving.detach().cpu(), field.detach().cpu()
        warped_moving_gt = warped_moving_gt.detach().cpu() if warped_moving_gt is not None else None
        field_gt = field_gt.detach().cpu() if field_gt is not None else None
        # #####################   FINAL RESOLUTION DEFORMATION FIELD   ############################
        if self.num_spatial_dims == 2:
            fig1 = visualise_results_2d(fixed, moving, warped_moving, field,
                                        transformed_image_gt=warped_moving_gt, field_gt=field_gt,
                                        title=f"{MODE_LUT[mode]} {modality}",
                                        metric_name=metric_name,
                                        error_warp_fix=error_warp_fix,
                                        error_mov_fix=error_mov_fix,
                                        error_warp_warpgt=error_warp_warpgt,
                                        error_warpgt_fix=error_warpgt_fix,
                                        vmax=vmax,
                                        cmap=cmap,
                                        )
            wandb.log({f'{mode}_images/{modality}': wandb.Image(fig1)})
        else:
            for i in range(3):
                c_axis = [j for j in range(3) if j != i]
                fig1 = visualise_results_2d(torch.index_select(fixed, i, shape[i]//2).squeeze(i),
                                            torch.index_select(moving, i, shape[i]//2).squeeze(i),
                                            torch.index_select(warped_moving, i, shape[i]//2).squeeze(i),
                                            torch.index_select(field, i, shape[i]//2).squeeze(i)[..., c_axis],
                                            transformed_image_gt=torch.index_select(warped_moving_gt, i, shape[i]//2).squeeze(i) if warped_moving_gt is not None else None,
                                            field_gt=torch.index_select(field_gt, i, shape[i]//2).squeeze(i)[..., c_axis] if field_gt is not None else None,
                                            title=f"{MODE_LUT[mode]} {modality} (along dim {i})",
                                            metric_name=metric_name,
                                            error_warp_fix=error_warp_fix,
                                            error_mov_fix=error_mov_fix,
                                            error_warp_warpgt=error_warp_warpgt,
                                            error_warpgt_fix=error_warpgt_fix,
                                            vmax=vmax,
                                            cmap=cmap,
                                            )
                wandb.log({f'{mode}_images/{modality}_dim{i}': wandb.Image(fig1)})

    def _log_multires_layerwise_def(self, batch, coord_components: Dict[str, List[torch.Tensor]], mode='train'):
        # #####################   MULTI-RESOLUTION LAYER-WISE DEFORMATION   ############################

        # Downsample GT images for each resolution
        m_ims = []
        f_ims = []
        m_im = batch["moving"][0:1]
        f_im = batch["fixed"][0:1]
        for i in range(len(coord_components["start"])):
            if i != 0:
                pool_size = self.network.encoder.pool_sizes[i - 1]
                m_im = self.pool_func(m_im, pool_size, ceil_mode=True)
                f_im = self.pool_func(f_im, pool_size, ceil_mode=True)
            _m_im = m_im.cpu().squeeze()
            _f_im = f_im.cpu().squeeze()
            if self.num_spatial_dims == 3:
                _m_im = _m_im[..., _f_im.shape[0]//2]
                _f_im = _f_im[..., _f_im.shape[0]//2]
            m_ims.append(_m_im)
            f_ims.append(_f_im)
        m_ims = m_ims[::-1]  # Reverse to coarse->fine (same as how coordinate components are ordered)
        f_ims = f_ims[::-1]

        # If available, downsample GT displacement for each resolution
        disp_gt = []
        if "disp" in batch:  # If a GT displacement exists, we will include it in the plot
            gt_d = batch["disp"][0:1]
            for i in range(len(coord_components["start"])):
                if i != 0:
                    pool_size = self.network.encoder.pool_sizes[i - 1]
                    gt_d = self.pool_func(gt_d.moveaxis(-1, 1), pool_size, ceil_mode=True).moveaxis(1, -1)
                _gt_d = gt_d.cpu().squeeze()
                disp_gt.append(_gt_d)
        disp_gt = disp_gt[::-1]

        # For each component in input dictionary, resample moving image at each resolution
        resampled_ims = {k: [] for k in coord_components.keys()}
        m_im = batch["moving"][0:1]
        for k, c_comp_per_res in coord_components.items():  # For each component
            for c_comp in c_comp_per_res:  # For each resolution coarse->fine
                if k in {"cross", "sib"}:
                    c_comp = c_comp[-1]
                c_comp = c_comp[0:1]
                warp_pred_im = resample(m_im, c_comp, interp_mode='bilinear')
                warp_pred_im = warp_pred_im.detach().cpu().squeeze()
                if self.num_spatial_dims == 3:
                    warp_pred_im = warp_pred_im[..., warp_pred_im.shape[0]//2]
                resampled_ims[k].append(warp_pred_im)
        # Create dictionary with plot's column titles
        im_dict = {
                   "Moving image": m_ims,
                   "Moving image\ninh": resampled_ims["inh"],
                   "Moving image\ninh+par": resampled_ims["par"],
                   "Moving image\ninh+par+cross ": resampled_ims["cross"],
                   "Moving image\ninh+par+cross+sib": resampled_ims["sib"],
                   "Fixed image": f_ims,
                   }

        # Create displacement fields we want to plot
        disp_inh = [(inh - s)[0].detach().cpu() for inh, s in zip(coord_components["inh"], coord_components["start"])]
        disp_par = [(par - s)[0].detach().cpu() for par, s in zip(coord_components["par"], coord_components["inh"])]
        disp_cross = [(cross[-1] - s)[0].detach().cpu() for cross, s in zip(coord_components["cross"], coord_components["par"])]
        disp_sib = [(sib[-1] - cross[-1])[0].detach().cpu() for sib, cross in zip(coord_components["sib"], coord_components["cross"])]
        disp_final = [(f - s)[0].detach().cpu() for f, s in zip(coord_components["final"], coord_components["start"])]
        if self.num_spatial_dims == 3:
            disp_gt = [i[..., i.shape[0] // 2, :-1] for i in disp_gt] if "disp" in batch else disp_gt
            disp_final = [i[..., i.shape[0] // 2, :-1] for i in disp_final]
            disp_inh = [i[..., i.shape[0] // 2, :-1] for i in disp_inh]
            disp_par = [i[..., i.shape[0] // 2, :-1] for i in disp_par]
            disp_cross = [i[..., i.shape[0] // 2, :-1] for i in disp_cross]
            disp_sib = [i[..., i.shape[0] // 2, :-1] for i in disp_sib]
        # Each displacement requires a corresponding starting coordinate
        coord_start = [i[0].detach().cpu() for i in coord_components["start"]]
        coord_inh = [i[0].detach().cpu() for i in coord_components["inh"]]
        coord_par = [i[0].detach().cpu() for i in coord_components["par"]]
        coord_cross = [i[-1][0].detach().cpu() for i in coord_components["cross"]]
        if self.num_spatial_dims == 3:
            coord_start = [i[..., i.shape[0] // 2, :-1] for i in coord_start]
            coord_inh = [i[..., i.shape[0] // 2, :-1] for i in coord_inh]
            coord_par = [i[..., i.shape[0] // 2, :-1] for i in coord_par]
            coord_cross = [i[..., i.shape[0] // 2, :-1] for i in coord_cross]

        # Place coordinate-displacement pairs into a dictionary with their corresponding plot column titles
        disp_dict = {
                   "Displacement\nstart->inh": (coord_start, disp_inh),
                   "Displacement\ninh->par": (coord_inh, disp_par),
                   "Displacement\npar->cross": (coord_par, disp_cross),
                   "Displacement\ncross->sib": (coord_cross, disp_sib),
                   "Displacement\nTotal": (coord_start, disp_final),
                   }
        if "disp" in batch:
            disp_dict = {"GT displacement": (coord_start, disp_gt), **disp_dict}
        fig4 = visualise_displ_components(disp_dict, im_dict)
        wandb.log({f'{mode}_images/displacements': fig4, "global_step": self.global_step})

    def training_step(self, batch, batch_idx):
        train_loss = self._step(batch, batch_idx, mode='train')
        return train_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self._step(batch, batch_idx, mode='val')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self._step(batch, batch_idx, mode='test')

    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        # save the metric results
        self.metric_reporter.summarise()
        self.metric_reporter.save_mean_std()
        self.metric_reporter.save_df()

        print(f'\νReport saved in: {self.hparams.current_dir}/analysis')
