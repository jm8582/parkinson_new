from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from load_pkl import PklLoader


class DataPreprocessor:
    def __init__(self, cfg, dset: PklLoader, before_intervals):
        self.dset = dset
        self.dset.reset_index(drop=True, inplace=True)
        self.dset["fog"] = self.dset["fog"].replace(range(len(cfg.fog_map)), cfg.fog_map)
        self.dset["act"] = self.dset["act"].replace(range(len(cfg.act_map)), cfg.act_map)
        max_interval = torch.ceil((before_intervals * (1 + cfg.max_pert)).sum()).long().item()
        self.dset_removed = self.remove_prev_prohibited_acts(self.dset, cfg.proh_acts, max_interval)

    def remove_prev_prohibited_acts(self, dset: pd.DataFrame, proh_acts: list, length: int):
        dset_act = dset["act"]
        dset_proh = dset_act.copy()
        dset_proh.values.fill(0)
        for i, a in enumerate(dset_act):
            if a in proh_acts:
                dset_proh[i : i + length + 1] = 1
        return dset[dset_proh == 0]

    def filter_dset(self, conds: List):
        return self.dset_removed[reduce(lambda acc, cur: acc & cur, [cond(self.dset_removed) for cond in conds])]


class TimeImgDset(Dataset):
    def __init__(
        self,
        cfg,
        dset_orig: pd.DataFrame,
        dset_removed: pd.DataFrame,
        before_intervals: list,
        max_pert: float,
        transform=False,
        targets=("fog", "act"),
    ):
        self.cfg = cfg
        self.dset_orig = dset_orig
        self.dset_removed = dset_removed
        self.before_intervals = before_intervals
        self.max_pert = max_pert
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.dset_removed)

    def __getitem__(self, idx):
        if self.transform:
            pert = np.random.uniform(low=1 - self.max_pert, high=1 + self.max_pert, size=self.cfg.in_dim)
        else:
            pert = np.ones(self.cfg.in_dim)
        perted_intervals = torch.cumsum(self.before_intervals * pert, 0).long()
        orig_idx = self.dset_removed.iloc[idx].name
        img = torch.from_numpy(np.stack(self.dset_orig.iloc[orig_idx - perted_intervals]["img_norm"].values).squeeze())
        img = self.transforms(img)
        tgt = torch.tensor(self.dset_orig.iloc[orig_idx][list(self.targets)]).squeeze()
        grp = torch.tensor(self.dset_orig.iloc[orig_idx]["group"])
        grp_by_8899 = torch.tensor(self.dset_orig.iloc[orig_idx]["group_by_8899"])
        return img, tgt, grp, grp_by_8899

    def transforms(self, img):
        trans = transforms.Compose(
            [
                transforms.RandomApply((transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),), p=0.5),
            ]
        )
        img = trans(img)
        if np.random.rand() > self.cfg.noise_prob:
            for _ in range(np.random.randint(1, self.cfg.max_num_noise + 1)):
                img[:, np.random.randint(15), np.random.randint(14)] = np.random.rand()
        img = img * np.random.uniform(1 - self.cfg.max_value_pert, 1 + self.cfg.max_value_pert)
        return img
