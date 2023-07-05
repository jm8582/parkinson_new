import io
import os

import pickle
import matplotlib.pyplot as plt
import model
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from load_data import DataPreprocessor, TimeImgDset
from PIL import Image

import wandb

MIN_INTERVAL = 0.02

cfg = {
    'proh_acts': [88, 99],
    'excluded_actions': [0, 1],
    # 'excluded_actions': [2,3,4,5,6,7,8],
    'train_ids': [1, 2, 3, 4, 6, 7, 9, 11, 12, 14, 15],
    'test_ids': [8, 10],
    # 'before_intervals': [0, 1, 2, 4, 8, 16, 32, 64],
    'before_intervals': [0, 1, 2, 4, 8, 16, 32, 64],
    'max_pert': 0.2,
    'fog_cls': ([0], [1, 2, 3]),
    'fog_map': (0, 0, 1, 1), 
    
    'model': 'res18',
    # output 차원, input 차원
    # 'model': ('res18', (4, 5)),
    # output 차원, input 차원
    
    'optimizer': 'Adam',
    'lr': 0.0005,
    'batch_size': 64,
    
    'gpu': 0,
    'n_ep': 1000,
    'test_interval': 1,
    'confusion_interval': 1,
    'model_save_interval': 1,
    
    'dset_path': '/workspace/data/data_max_neigh.pkl',
}

cfg['input_dim'] = len(cfg['before_intervals'])
cfg['output_dim'] = len(cfg['fog_cls'])

default_cond = []
if cfg['excluded_actions']:
    default_cond.append(lambda x: ~x['action'].isin(cfg['excluded_actions']))
train_conds = default_cond + [lambda x: x['id'].isin(cfg['train_ids'])]
test_conds = default_cond + [lambda x: x['id'].isin(cfg['test_ids'])]
    
with open(cfg['dset_path'], 'rb') as f:
    dset_pd = pickle.load(f)
d_pre = DataPreprocessor(cfg, dset_pd, cfg['before_intervals'])
dset_pd = d_pre.dset
dset_train_pd, dset_test_pd = d_pre.filter_dset(train_conds), d_pre.filter_dset(test_conds)
dset_train = TimeImgDset(dset_pd, dset_train_pd, cfg['before_intervals'], cfg['max_pert'], transform=True)
dset_test = TimeImgDset(dset_pd, dset_test_pd, cfg['before_intervals'], cfg['max_pert'], transform=False)
    
cls_sample_count = dset_train_pd['target'].value_counts().values
cls_weight = 1 / cls_sample_count
sample_weight = np.array([cls_weight[t] for t in dset_train_pd['target']])
sampler = WeightedRandomSampler(sample_weight, 1280, replacement=True)
dl_train = DataLoader(dset_train, batch_size=128, sampler=sampler)
    
cls_sample_count = dset_test_pd['target'].value_counts().values
cls_weight = 1 / cls_sample_count
sample_weight = np.array([cls_weight[t] for t in dset_test_pd['target']])
sampler = WeightedRandomSampler(sample_weight, len(dset_test), replacement=True)
dl_test = DataLoader(dset_test, batch_size=128, sampler=sampler)

group_map = {}
idxs = dset_pd[dset_pd['group'].shift() != dset_pd['group']].index

group_map = {}
idxs = dset_pd[dset_pd['group'].shift() != dset_pd['group']].index
for i in range(len(idxs) - 1):
    if dset_pd.iloc[idxs[i]]['action'] not in (88, 99):
        try:
            group_map[dset_pd.iloc[idxs[i+1]]['group']] = dset_pd.iloc[idxs[i]]['group']
        except:
            pass
        
dset_pd['group_by_8899'] = dset_pd['group'].replace(group_map)

with open(cfg['dset_path'], 'wb') as f:
    pickle.dump(dset_pd, f)