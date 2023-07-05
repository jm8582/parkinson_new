import io
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchmetrics import Specificity

import model
import wandb
from cosine_anealing_with_warmup import CosineAnnealingWarmUpRestarts


def specificity_score(y_true, y_pred, n_cls, average="micro"):
    specificity = Specificity(task="multiclass", average=average, num_classes=n_cls)
    return specificity(y_pred, y_true)


class LitNet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.maxs = defaultdict(int)
        self.val_dl_dict = {
            0: "val_sample",
            1: "val_sample_noisy",
            2: "val_full",
            3: "val_full_noisy",
        }

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optim)(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer,
                T_0=self.cfg.T0,
                T_mult=self.cfg.T_mult,
                eta_max=max(self.cfg.max_lr, self.cfg.lr * 10),
                T_up=self.cfg.T_up,
                gamma=self.cfg.gamma,
            )
        elif self.cfg.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max(self.cfg.max_lr, self.cfg.lr * 10),
                steps_per_epoch=self.cfg.step_per_epoch,
                epochs=self.cfg.max_epoch,
            )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx, dl_idx):
        return self.step(batch, batch_idx, mode=self.val_dl_dict[dl_idx])

    def step(self, batch, batch_idx, mode="train"):
        x, y, _, _ = batch
        if "fog" in self.cfg.targets and "act" in self.cfg.targets:
            y_fog, y_act = y[:, 0].squeeze(), y[:, 1].squeeze()
        elif "fog" in self.cfg.targets:
            y_fog, y_act = y.squeeze(), None
        elif "act" in self.cfg.targets:
            y_fog, y_act = None, y.squeeze()

        if "act" in self.cfg.targets and np.random.rand() < self.cfg.horizontal_flip_prob:
            x = torchvision.transforms.functional.hflip(x)
            y_act[y_act == 2], y_act[y_act == 3] = 3, 2

        logs = {}
        return_dict = {}
        for tgt_name in self.cfg.targets:
            if tgt_name == "fog":
                tgt, net = y_fog, self.fog_net
            elif tgt_name == "act":
                tgt, net = y_act, self.act_net

            logits = net(x)
            loss = F.cross_entropy(logits, tgt)
            preds = logits.argmax(dim=1)
            acc = (preds == tgt).float().mean()

            logs.update({f"{mode}_{tgt_name}_loss": loss, f"{mode}_{tgt_name}_acc": acc})
            self.log_dict(logs, on_epoch=True)

            return_dict[f"pred_{tgt_name}"] = preds
            return_dict[f"tgt_{tgt_name}"] = tgt
        if mode == "train":
            if "fog" in self.cfg.targets and "act" in self.cfg.targets:
                return_dict["loss"] = (1 - self.cfg.act_w) * logs["train_fog_loss"] + self.cfg.act_w * logs["train_act_loss"]
            elif "fog" in self.cfg.targets:
                return_dict["loss"] = logs["train_fog_loss"]
            elif "act" in self.cfg.targets:
                return_dict["loss"] = logs["train_act_loss"]

        return return_dict

    def training_epoch_end(self, validation_step_outputs):
        self.epoch_end(validation_step_outputs, mode="train")

    def validation_epoch_end(self, validation_step_outputs):
        for i, output in enumerate(validation_step_outputs):
            self.epoch_end(output, mode=self.val_dl_dict[i])

    def epoch_end(self, outputs, mode="train"):
        outputs_dict = {}
        for k in outputs[0].keys():
            if outputs[0][k].ndim:
                outputs_dict[k] = torch.concat([x[k] for x in outputs]).to("cpu")

        for tgt_name in self.cfg.targets:
            tgt, preds = outputs_dict[f"tgt_{tgt_name}"], outputs_dict[f"pred_{tgt_name}"]
            confusion = confusion_matrix(tgt, preds)
            self.log_confusion(confusion, sorted(torch.concat((tgt, preds)).unique().numpy()), f"{mode}_{tgt_name}")

            def _logger(name, score):
                for i, s in enumerate(score):
                    self.log(f"{mode}_{tgt_name}{i}_{name}", s)

            n_cls = int(max(self.cfg.fog_map)) + 1 if tgt_name == "fog" else int(max(self.cfg.act_map)) + 1
            self.log(f"{mode}_{tgt_name}_acc", accuracy_score(tgt, preds))
            _logger("prec", precision_score(tgt, preds, average=None))
            _logger("recall", recall_score(tgt, preds, average=None))
            _logger("f1", f1_score(tgt, preds, average=None))
            _logger("spec", specificity_score(tgt, preds, n_cls, average=None))

        if "fog" in self.cfg.targets and "act" in self.cfg.targets:
            tgt = outputs_dict["tgt_fog"] * 10 + outputs_dict["tgt_act"]
            preds = outputs_dict["pred_fog"] * 10 + outputs_dict["pred_act"]
            confusion = confusion_matrix(tgt, preds)
            self.log_confusion(confusion, sorted(torch.concat((tgt, preds)).unique().numpy()), f"{mode}_fog+act")

    def log_confusion(self, cm, ticklabels, ylabel="fog"):
        s = sn.heatmap(cm, annot=True, fmt="d", square=True, cmap="Blues", xticklabels=ticklabels, yticklabels=ticklabels)
        s.set(xlabel="pred", ylabel=ylabel)
        fig = plt.gcf()
        self.log_fig(fig, ylabel)

    def log_fig(self, fig, name):
        if "wandb" in self.cfg.logger:
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            with Image.open(buf) as image:
                self.logger.experiment.log({name: wandb.Image(image)})
            plt.close("all")
        else:
            print(f"image can be logged only with wandb logger. {name=}")


class LitResNet(LitNet):
    def __init__(self, **kwargs):
        super().__init__()

    def init_net(self):
        net = model.net_wrapper(
            getattr(model, self.cfg.net),
            block_channel_starts=self.cfg.block_channel_starts,
            input_kernel_size=self.cfg.in_dim,
        )
        if "fog" in self.cfg.targets:
            self.fog_net = net(n_cls=self.cfg.fog_dim)
        if "act" in self.cfg.targets:
            self.act_net = net(n_cls=self.cfg.act_dim)

        if "fog" in self.cfg.targets and "act" in self.cfg.targets:
            self.act_net.conv1 = self.fog_net.conv1
            self.act_net.bn1 = self.fog_net.bn1
            self.act_net.layer1 = self.fog_net.layer1
            self.act_net.layer2 = self.fog_net.layer2

        return
