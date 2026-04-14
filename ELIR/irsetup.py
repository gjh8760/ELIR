from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
import pytorch_lightning as L
import torch
from ELIR.metrics import MetricEval
from ELIR.training.losses import get_loss
from torchvision.utils import save_image, make_grid
from ELIR.training.ema_timm import ModelEMA
import os
import torch.nn.functional as F
from ELIR.utils import ImageSpliterTh
import math



class IRSetup(L.LightningModule):
    def __init__(self, model, fm_cfg={}, optimizer=None, scheduler=None, tmodel=None,
                 ema_decay=None, eval_cfg=None, run_dir=None, save_images=True):
        super().__init__()
        self.model = model
        self.fm_cfg = fm_cfg
        self.eval_cfg = eval_cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = eval_cfg.get("metrics",[])
        self.tmodel = tmodel
        if torch.cuda.is_available():
            self.acc_device = torch.device(torch.cuda.current_device())
        elif torch.backends.mps.is_available():
            self.acc_device = torch.device('mps')
        else:
            self.acc_device = torch.device('cpu')
        self.metric_evals = [MetricEval(metric, self.acc_device, run_dir) for metric in self.metrics]
        self.train_loss = []
        self.ema = None
        if ema_decay:
            self.ema = ModelEMA(model, device=self.acc_device, decay=ema_decay)
        self.samples_dir = None
        if run_dir and save_images:
            self.samples_dir = os.path.join(run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)  # run folder
            self.samples = []

        # Full eval-set image logging (LQ | Pred | HQ triptych per sample)
        self.log_images = bool(eval_cfg.get("log_images", False)) if eval_cfg else False
        self.max_log_images = int(eval_cfg.get("max_log_images", 32)) if eval_cfg else 32
        self._val_lq = []
        self._val_pred = []
        self._val_hq = []

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        # Loss function
        loss = get_loss(self.model, x_hq, x_lq, self.fm_cfg, self.tmodel)

        self.train_loss.append(loss)
        if batch_idx % 5:
            self.log("train_loss", torch.mean(torch.Tensor(self.train_loss)).item(), logger=True, prog_bar=True)
            self.train_loss.clear()
        return loss

    def compute_metrics(self, x_hq_hat, x_hq):
        for metric_eval in self.metric_evals:
            metric_eval.compute(x_hq_hat, x_hq)

    def save_samples(self, current_epoch):
        save_image(torch.concat(self.samples,dim=0), os.path.join(self.samples_dir,"epoch_"+str(current_epoch)+".png"))

    def infer(self, x):
        if self.ema:
            return self.ema.model.inference(x)
        else:
            return self.model.inference(x)

    def validation_step(self, batch, batch_idx):
        x_lq, y = batch
        chop = self.eval_cfg.get("chop", None)
        if chop:
            sf = chop.get("sf", 4)
            upscale = chop.get("upscale", 4)
            chop_size = chop.get("chop_size", 256)
            chop_stride = chop.get("chop_stride", 224)
            chop_size_h = chop.get("chop_size_h", chop_size)
            chop_size_w = chop.get("chop_size_w", chop_size)
            chop_stride_h = chop.get("chop_stride_h", chop_stride)
            chop_stride_w = chop.get("chop_stride_w", chop_stride)
            patch_spliter = ImageSpliterTh(x_lq, pch_size=(chop_size_h, chop_size_w),
                                           stride=(chop_stride_h, chop_stride_w), sf=sf, extra_bs=1)
            for patch, index_infos in patch_spliter:
                patch_h, patch_w = patch.shape[2:]
                flag_pad = False
                if not (patch_h % 64 == 0 and patch_w % 64 == 0):
                    flag_pad = True
                    pad_h = (math.ceil(patch_h / 64)) * 64 - patch_h
                    pad_w = (math.ceil(patch_w / 64)) * 64 - patch_w
                    patch = F.pad(patch, pad=(0, pad_w, 0, pad_h), mode='reflect')
                pad_patch_h, pad_patch_w = patch.shape[2:]
                patch = F.interpolate(patch, size=(upscale*pad_patch_h, upscale*pad_patch_w), mode='bicubic')
                y_hat = self.infer(patch)
                if flag_pad:
                    y_hat = y_hat[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(y_hat, index_infos)
            y_hat = patch_spliter.gather()
        else:
            y_hat = self.infer(x_lq)

        if self.current_epoch > 0 and batch_idx < 2 and self.samples_dir and torch.cuda.current_device()==0:
            self.samples.append(y_hat[:2,...]) # save 2 images
            if len(self.samples) == 2: # save 2 batches
                self.save_samples(self.current_epoch)
                self.samples.clear()

        # Collect full eval triptychs (LQ | Pred | HQ) for on_validation_epoch_end.
        if (self.log_images
                and not self.trainer.sanity_checking
                and self.trainer.is_global_zero
                and self.samples_dir is not None
                and len(self._val_pred) < self.max_log_images):
            take = min(y_hat.shape[0], self.max_log_images - len(self._val_pred))
            self._val_lq.append(x_lq[:take].detach().float().cpu().clamp(0, 1))
            self._val_pred.append(y_hat[:take].detach().float().cpu().clamp(0, 1))
            self._val_hq.append(y[:take].detach().float().cpu().clamp(0, 1))

        self.compute_metrics(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('global_step', self.global_step)
        for metric_eval in self.metric_evals:
            result = metric_eval.get_final().item()
            self.log(metric_eval.metric, result, sync_dist=True, prog_bar=True)

        # Save full eval triptychs to disk and push them to the logger.
        if (self.log_images
                and not self.trainer.sanity_checking
                and self.trainer.is_global_zero
                and len(self._val_pred) > 0
                and self.samples_dir is not None):
            self._save_and_log_val_samples()
        self._val_lq.clear()
        self._val_pred.clear()
        self._val_hq.clear()

        torch.cuda.empty_cache()

    def _save_and_log_val_samples(self):
        epoch_dir = os.path.join(self.samples_dir, "epoch_{:04d}".format(self.current_epoch))
        os.makedirs(epoch_dir, exist_ok=True)

        grids, captions = [], []
        idx = 0
        for lq_b, pr_b, hq_b in zip(self._val_lq, self._val_pred, self._val_hq):
            for i in range(pr_b.shape[0]):
                triptych = make_grid(
                    torch.stack([lq_b[i], pr_b[i], hq_b[i]], dim=0),
                    nrow=3, padding=2, pad_value=1.0)
                save_image(triptych, os.path.join(epoch_dir, "{:03d}.png".format(idx)))
                grids.append(triptych)
                captions.append("{:03d} LQ|Pred|HQ".format(idx))
                idx += 1

        if self.logger is not None and hasattr(self.logger, "log_image"):
            try:
                self.logger.log_image(key="val/samples", images=grids, caption=captions)
            except Exception as e:
                print("Warn: logger.log_image failed: {}".format(e))

    def on_save_checkpoint(self, checkpoint):
        src = self.ema.model if self.ema else self.model
        # Phase-2 ElirRetinex: sub-branches have distinct keys.
        if hasattr(src, "R_fmir"):
            checkpoint['state_dict_R_fmir'] = src.R_fmir.state_dict()
            checkpoint['state_dict_R_mmse'] = src.R_mmse.state_dict()
            checkpoint['state_dict_R_enc']  = src.R_enc.state_dict()
            checkpoint['state_dict_R_dec']  = src.R_dec.state_dict()
            checkpoint['state_dict_I_fmir'] = src.I_fmir.state_dict()
            checkpoint['state_dict_I_mmse'] = src.I_mmse.state_dict()
            # decomposer is frozen; reconstructed from config on load.
        else:
            # Phase-1 Elir: unchanged.
            if hasattr(src, "fmir"):
                checkpoint['state_dict_fmir'] = src.fmir.state_dict()
            if hasattr(src, "mmse"):
                checkpoint['state_dict_mmse'] = src.mmse.state_dict()
            if hasattr(src, "enc"):
                checkpoint['state_dict_enc'] = src.enc.state_dict()
            if hasattr(src, "dec"):
                checkpoint['state_dict_dec'] = src.dec.state_dict()
        return checkpoint

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [self.scheduler]
