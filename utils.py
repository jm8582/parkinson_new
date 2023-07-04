from glob import glob

import numpy as np
import pandas as pd


def raw2img(data: np.ndarray):
    zero_idx = [0, 5, 93, 95, 98, 99]
    ldata = np.insert(data[..., :99], zero_idx, 0, axis=-1).reshape(data.shape[:-1] + (15, 7))
    rdata = np.insert(data[..., 99:], zero_idx, 0, axis=-1).reshape(data.shape[:-1] + (15, 7))
    left = np.flip(ldata, axis=(-1, -2))
    right = np.flip(rdata, axis=(-2,))
    return np.concatenate((left, right), axis=-1)


def img2raw(img: np.ndarray):
    # split the image into left and right
    left, right = np.split(img, 2, axis=-1)
    
    # reverse the flip operations
    left = np.flip(left, axis=(-1, -2))
    right = np.flip(right, axis=(-2,))
    
    # reshape back to original shape
    left = left.reshape(img.shape[:-2] + (-1,))
    right = right.reshape(img.shape[:-2] + (-1,))
    
    # remove the zero-inserted indices
    zero_idx = [0, 6, 95, 98, 102, 104]
    left = np.delete(left, zero_idx, axis=-1)
    right = np.delete(right, zero_idx, axis=-1)
    
    # concatenate the left and right data
    raw = np.concatenate((left, right), axis=-1)
    return raw

def np2pd(arr: np.ndarray) -> pd.DataFrame:
    # define column and index names
    columns = list(range(1, 100)) + [f"{i}.1" for i in range(1, 100)]
    index = list(range(1, len(arr) + 1))

    # create dataframe
    return pd.DataFrame(arr, columns=columns, index=index)

def normalize(arr: np.ndarray):
    return (arr - arr.min()) / (arr.max() - arr.min())
