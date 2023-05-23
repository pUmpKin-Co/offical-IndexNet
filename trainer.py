import torch
from functools import partial
import pytorch_lightning as pl
from typing import Callable, Sequence
from IndexNetModel.IndexNetContrastMo import IndexNet, conv_loss, linear_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class IndexNetTrainer(pl.LightningModule):
    def __init__(self, config):
        super(IndexNetTrainer, self).__init__()
        self.optimizer = config.schedule.optimizer
        self.lr = config.schedule.base_weight_lr
        self.weight_decay = config.schedule.weight_decay
        self.scheduler = config.schedule.schedule_name
        self.max_epochs = config.schedule.epochs
        self.loss_weight = config.network.loss_weight_alpha
        self.spatial_dimension = config.network.spatial_dimension
        self.with_global = config.with_global

        if self.scheduler is not None:
            self.min_lr = config.schedule.min_lr

        if self.scheduler == "warmup_cosine":
            self.warmup_epochs = config.schedule.warmup_epochs
            self.warmup_start_lr = config.schedule.warmup_lr
        elif self.scheduler == "step":
            self.lr_decay_steps = config.schedule.decay_rate

        self.model = IndexNet(config)

    def configure_optimizers(self):

        idxs_no_scheduler = [
            i for i, m in enumerate(self.model.learnable_params) if m.pop("static_lr", False)
        ]

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        optimizer = optimizer(
            self.model.learnable_params,
            weight_decay=self.weight_decay
        )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr * 10] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]


    def on_train_start(self):
        self.last_step = 0

    def forward(self, img1, img2):
        return self.model(img1, img2)

    def training_step(self, imges_dict, batch_idx):
        img1 = imges_dict["img1"]
        img2 = imges_dict["img2"]

        loss = 0

        if self.with_global:
            pred1, pred2, mo_feat1, mo_feat2, linear_pred_1, linear_pred_2, mo_linear_feat1, mo_linear_feat2 = self(img1, img2)
            for stage, (q1, k2) in enumerate(zip(linear_pred_1, mo_linear_feat2)):
                loss += linear_loss(q1, k2)

            for stage, (q2, k1) in enumerate(zip(linear_pred_2, mo_linear_feat1)):
                loss += linear_loss(q2, k1)
        else:
            pred1, pred2, mo_feat1, mo_feat2 = self(img1, img2)

        mask1 = imges_dict["mask1"]
        mask2 = imges_dict["mask2"]


        for stage, (feature_q, feature_k) in enumerate(zip(pred1, mo_feat2)):
            loss += conv_loss(feature_q, feature_k, mask1, mask2,
                              self.spatial_dimension[stage], self.temperature) * self.loss_weight[stage]

        for stage, (feature_q, feature_k) in enumerate(zip(pred2, mo_feat1)):
            loss += conv_loss(feature_q, feature_k, mask2, mask1,
                              self.spatial_dimension[stage], self.temperature) * self.loss_weight[stage]

        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.trainer.global_step > self.last_step:
            momentum_pairs = self.model.momentum_pairs
            for mp in momentum_pairs:
                self.model.ema.update(*mp)

            self.log("tau", self.model.ema.cur_tau)

            cur_step = self.trainer.global_step
            if self.trainer.accumulate_grad_batches:
                cur_step = cur_step * self.trainer.accumulate_grad_batches
            self.model.ema.update_tau(cur_step=cur_step,
                                      max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs)

        self.last_step = self.trainer.global_step





