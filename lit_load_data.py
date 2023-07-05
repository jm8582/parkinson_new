import pickle

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb
from load_data import DataPreprocessor, TimeImgDset

# serial_weight = {
#     0: 1,
#     1: 1,
#     2: 1,
#     3: 0.2,
#     10: 1,
#     11: 1,
#     12: 1,
#     13: 0.2,
# }

# serial_weight = {
#     0: 1,
#     1: 1.2,
#     2: 1,
#     10: 2,
#     11: 2.4,
#     12: 2,
# }


class ImgDataModule(LightningDataModule):
    def __init__(self, cfg, before_intervals):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.before_intervals = before_intervals

    def setup(self, stage):
        default_cond = []
        if self.cfg.excluded_acts:
            default_cond.append(lambda x: ~x["act"].isin(self.cfg.excluded_acts))
        train_conds = default_cond + [lambda x: x["id"].isin(self.cfg.train_ids)]
        test_conds = default_cond + [lambda x: x["id"].isin(self.cfg.test_ids)]

        path = self.cfg.dset_path + "_swing.pkl" if self.cfg.swing_neigh else self.cfg.dset_path + ".pkl"
        print(type(self.cfg.swing_neigh))
        print(self.cfg.swing_neigh)
        print(path)
        with open(path, "rb") as f:
            dset_pd = pickle.load(f)
        d_pre = DataPreprocessor(self.cfg, dset_pd, self.before_intervals)
        self.dset_train_pd, self.dset_test_pd = d_pre.filter_dset(train_conds), d_pre.filter_dset(test_conds)

        self.dset_train = TimeImgDset(
            self.cfg,
            d_pre.dset,
            self.dset_train_pd,
            self.before_intervals,
            self.cfg.max_pert,
            transform=True,
            targets=self.cfg.targets,
        )
        self.dset_test = TimeImgDset(
            self.cfg,
            d_pre.dset,
            self.dset_test_pd,
            self.before_intervals,
            self.cfg.max_pert,
            transform=False,
            targets=self.cfg.targets,
        )
        self.dset_test_noisy = TimeImgDset(
            self.cfg,
            d_pre.dset,
            self.dset_test_pd,
            self.before_intervals,
            self.cfg.max_pert,
            transform=True,
            targets=self.cfg.targets,
        )

        if "wandb" in self.cfg.logger:
            wandb.log({"train_size": len(self.dset_train), "test_size": len(self.dset_test)})

    def weighted_dataloader(self, dset, dset_pd, step_per_epoch):
        dset_pd["serialized"] = dset_pd["fog"] * 10 + dset_pd["act"]
        cls_sample_count = dset_pd["serialized"].value_counts()
        cls_weight = dict(1 / cls_sample_count)
        sample_weight = np.array([cls_weight[t] for t in dset_pd["serialized"]])
        sampler = WeightedRandomSampler(sample_weight, self.batch_size * step_per_epoch, replacement=True)
        dl = DataLoader(dset, batch_size=self.batch_size, sampler=sampler, num_workers=16)
        return dl

    def train_dataloader(self):
        """returns training dataloader"""
        return self.weighted_dataloader(self.dset_train, self.dset_train_pd, step_per_epoch=self.cfg.step_per_epoch)

    def val_dataloader(self):
        """returns validation dataloader"""
        sample = self.weighted_dataloader(self.dset_test, self.dset_test_pd, step_per_epoch=self.cfg.step_per_epoch)
        sample_noisy = self.weighted_dataloader(self.dset_test_noisy, self.dset_test_pd, step_per_epoch=self.cfg.step_per_epoch)
        full = DataLoader(self.dset_test, batch_size=self.batch_size, shuffle=False, num_workers=16)
        full_noisy = DataLoader(self.dset_test_noisy, batch_size=self.batch_size, shuffle=False, num_workers=16)
        return [sample, sample_noisy, full, full_noisy]

    def test_dataloader(self):
        """returns test dataloader"""
        return DataLoader(self.dset_test, batch_size=self.batch_size, shuffle=False, num_workers=16)
