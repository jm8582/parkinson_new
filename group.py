import pandas as pd
import pickle
from itertools import product

cfg = {
    'prohibited_actions': [88, 99],
    'excluded_actions': [2, 3, 4, 5, 6, 7, 8],
    'train_ids': [1, 2, 3, 4, 6, 7, 9, 11, 12, 14, 15],
    'test_ids': [8, 10],
    'before_intervals': [5, 5, 5, 5, 5],
    'fog_cls': ([0], [1, 2, 3]),
    
    'model': 'res18',
    'output_dim': 4,
    'input_dim': 6,
    # output 차원, input 차원
    
    'optimizer': 'Adam',
    'lr': 0.0005,
    'batch_size': 32,
    
    'gpu': 0,
    'n_iter': 500,
    'log_interval': 10,
    'confusion_interval': 10,
    
    'dset_path': "/workspace/data/data_max_neigh_swing.pkl",
    
    'seed': 1
}

with open(cfg['dset_path'], 'rb') as f:
    dset = pickle.load(f)
    
dset.reset_index(drop=True, inplace=True)
d_labels = dset[['trial', 'id', 'action', 'fog']].sort_values(['id', 'trial']).copy()
d_labels['target'] = d_labels['fog'].isin(cfg['fog_cls'][1]).astype(int)
group_by_target = (d_labels['target'].shift() != d_labels['target']).cumsum()
d_labels['group'] = d_labels['id']*100000 + d_labels['trial'] * 1000 + group_by_target
dset['group'] = d_labels['group']
dset['group_by_8899'] = dset['group']
dset = dset.sort_values(['id', 'trial', 'time'])
with open(cfg['dset_path'], 'wb') as f:
    pickle.dump(dset, f)