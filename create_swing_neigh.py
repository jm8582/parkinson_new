import os
import pickle
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy

import utils
from find_time_avg_at_valley import find_time_avg_at_valley
from load_pkl import loader


def create_neighbored_img(img: np.ndarray, err_pos: np.ndarray, r: int = 1):
    img = img.copy()
    img[:, list(map(tuple, err_pos.T))] = 0
    err_pos_set = set(map(tuple, err_pos))
    for err_pos_tup in sorted(err_pos_set.copy(), key=lambda x: x[0] * 100 + x[1]):
        x, y = err_pos_tup
        neighbors_pos = {
            (x + i, y + j)
            for i, j in product(range(-r, r + 1), repeat=2)
            if x + i >= 0 and y + j >= 0 and x + i < img.shape[-2] and y + j < img.shape[-1]
        }

        idx = list(map(tuple, np.array(list(neighbors_pos)).T))
        neighbors = img[np.arange(img.shape[0])[:, None], idx[0], idx[1]]
        err_in_neigh_pos = neighbors_pos & err_pos_set
        idx = list(map(tuple, np.array(list(err_in_neigh_pos)).T))
        errs = img[np.arange(img.shape[0])[:, None], idx[0], idx[1]]

        img[:, x, y] = (neighbors.sum(axis=1) - errs.sum(axis=1)) / (neighbors.shape[1] - errs.shape[1])
        err_pos_set.remove(err_pos_tup)
    return img


def find_nonzero_position(img: np.ndarray):
    return np.stack(img.nonzero()).T


if __name__ == "__main__":
    # save neighbored
    data = loader.data_id_trial_type
    raws = loader.data_max_neigh
    r = 2

    for id_ in raws.keys():
        trials = data[id_].keys()
        for trial_i, trial in enumerate(trials):
            d = data[id_][trial]
            raw = raws[id_][trial]
            img = utils.raw2img(raw.values)
            left_img = img[..., :7]
            right_img = img[..., 7:]
            tag = d.get("tag")

            l_time_avg_at_valley, r_time_avg_at_valley = find_time_avg_at_valley(raw, tag)

            err_img = utils.raw2img(np.concatenate([l_time_avg_at_valley, r_time_avg_at_valley]))
            err_left = err_img[..., :7]
            err_right = err_img[..., 7:]

            neighbored_left = create_neighbored_img(left_img, find_nonzero_position(err_left), r=r)
            neighbored_right = create_neighbored_img(right_img, find_nonzero_position(err_right), r=r)
            neighbored_img = np.concatenate([neighbored_left, neighbored_right], axis=-1)
            neighbored_raw = utils.img2raw(neighbored_img)
            neighbored_raw_pd = utils.np2pd(neighbored_raw)

            path = f"/workspace/data/pkl/{id_}/swing_neigh_r{r}"
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/{trial}.pkl", "wb") as f:
                pickle.dump(neighbored_raw_pd, f)
