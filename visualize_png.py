import datetime
import os
import pickle
from collections import defaultdict
from itertools import product

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

from utils import raw2img
from load_pkl import PklLoader

insole_for_id = {
    1: "U-1053",
    2: "U-1053",
    3: "W-2127",
    4: "W-1745",
    5: "W-1745",
    6: "W-1745",
    7: "W-1745",
    8: "W-2693",
    9: "U-1053",
    10: "W-2693",
    11: "U-1389",
    12: "V-2187",
    13: "V-2187",
    14: "W-2693",
}

date_for_id = {
    1: "04.04",
    2: "04.14",
    3: "04.22",
    4: "05.18",
    5: "05.19",
    6: "05.20",
    7: "05.24",
    8: "06.10",
    9: "07.26",
    10: "08.01",
    11: "08.02",
    12: "08.05",
    13: "08.09",
    14: "08.10",
}


def load_id_trial_dict(start_id=1, end_id=18, trial_count=4, base_path="./data/pkl", category="raw", neighbor=[1, 2, 3]):
    id_trial = defaultdict(dict)

    for id_, trial in product(range(start_id, end_id), range(trial_count + 1)):
        file_path = f"{base_path}/{id_}/{category}/{trial}.pkl"

        data = []
        if os.path.exists(file_path):
            if category == "raw":
                with open(file_path, "rb") as f:
                    data.append(pickle.load(f))
                for i in neighbor:
                    n_path = file_path.replace("raw", f"n{i}")
                    if os.path.exists(n_path):
                        with open(n_path, "rb") as f:
                            data.append(pickle.load(f))

                id_trial[id_][trial] = np.stack(data)
            else:
                with open(file_path, "rb") as f:
                    id_trial[id_][trial] = pickle.load(f)

    return id_trial


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)

    normalized_data = (data - min_val) / (max_val - min_val) * 100
    return normalized_data


def plot_heatmap_with_features(data: np.ndarray, title: str, ax: plt.Axes, recent_cop: np.ndarray = None):
    height, width = data.shape

    # Check if all values in data are zero
    y_center, x_center = (height / 2, width / 2) if np.all(data == 0) else ndimage.center_of_mass(data)

    mean = np.mean(data)

    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=100)
    ax.scatter(x_center, y_center, color="red", marker="x", linewidths=4, s=200)
    ax.set_title(f"{title}\nMean: {mean:0>5.2f}\nCoP: ({x_center:0>5.2f}, {y_center:0>5.2f})")

    if recent_cop is not None:
        colormap = cm.viridis
        viridis_to_red = mcolors.LinearSegmentedColormap.from_list("viridis_to_red", [colormap(0), "red"])
        color_indices = np.logspace(0, 1, len(recent_cop)) / 10

        alpha_vals = np.linspace(0, 1.0, len(recent_cop))
        for i, (alpha, cop) in enumerate(zip(alpha_vals, recent_cop)):
            y, x = cop
            color = viridis_to_red(color_indices[i])
            ax.scatter(x, y, color="red", marker="o", s=80, alpha=alpha)
            ax.text(x, y, f"{len(recent_cop) - i}", fontsize=12, color=color)

            if i > 0:
                prev_cop = recent_cop[i - 1]
                ax.plot(
                    [prev_cop[1], cop[1]],
                    [prev_cop[0], cop[0]],
                    color="cyan",
                    alpha=alpha,
                    linewidth=1.5,
                )

    return im


def plot_action_fog(ax: plt.Axes, fogs: np.ndarray, acts: np.ndarray, fnum: int):
    action_idx_map = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 99, 88], [1, 2, 3, 7, 6, 5, 4, 9, 8, 10, 0]))

    action_tags = ["sit", "stand", "start", "straight", "L90", "R90", "L180", "R180", "L360", "R360", "unexpected"]
    fog_tags = ["Normal", "Non-magnetic", "Magnetic", "Freeze"]

    ax.plot(np.arange(len(fogs)) * 0.02, fogs, "b-", label="fog")
    ax.set_yticks(range(len(fog_tags)))
    ax.set_yticklabels(fog_tags, color="blue", fontsize=12)

    ax2 = ax.twinx()
    ax2.plot(np.arange(len(fogs)) * 0.02, np.vectorize(action_idx_map.get)(acts), "r-")
    ax2.set_ylim(0, 11)
    ax2.set_yticks(range(len(action_tags)))
    ax2.set_yticklabels(action_tags, color="red", fontsize=12)
    ax2.grid("on")

    ax.plot(fnum * 0.02, fogs[fnum], "bo", markersize=8)
    ax2.plot(fnum * 0.02, action_idx_map[acts[fnum]], "ro", markersize=8)

    return


def process_pressure_data(
    data: np.ndarray,
    fogs: np.ndarray,
    acts: np.ndarray,
    id_: int,
    trial: int,
    fnum: int,
    cop: np.ndarray = None,
    lf_cols=slice(0, 7),
    rf_cols=slice(7, 14),
):
    lf_data = data[..., lf_cols]
    rf_data = data[..., rf_cols]

    neighbors = len(data)
    fig, axes = plt.subplots(1, neighbors * 2, figsize=(10 * neighbors, 18), facecolor="white")

    recent_duration = 50
    if cop is not None:
        recent_cop = cop[:, fnum - recent_duration + 1 : fnum + 1, ...]
        l_recent_cop, r_recent_cop = recent_cop[..., :2], recent_cop[..., 2:]
    else:
        l_recent_cop, r_recent_cop = None, None

    for i, (lf, rf, lc, rc) in enumerate(zip(lf_data, rf_data, l_recent_cop, r_recent_cop)):
        im1 = plot_heatmap_with_features(lf, f"n{i} left", axes[i], recent_cop=lc)
        im2 = plot_heatmap_with_features(rf, f"n{i} right", axes[i + neighbors], recent_cop=rc)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

    # Create a new axis for colorbar
    cbar_ax = fig.add_axes([0.15, 0.23, 0.7, 0.015])
    fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")

    # Create a new axis for the plot_action_fog plot
    action_fog_ax = fig.add_axes([0.15, 0.05, 0.7, 0.16])
    plot_action_fog(action_fog_ax, fogs, acts, fnum)

    time = fnum * 0.02
    fig.text(
        0.2,
        0.77,
        f"ID: {id_:02d} | Trial: {trial:02d} | Insole: {insole_for_id.get(id_, 'unknown')} | Date: {date_for_id.get(id_, 'unknown')}\n"
        f"FoG: {fogs[fnum]} | Act: {acts[fnum]}\n"
        f"Frame: {fnum:05d} | Time: {str(datetime.timedelta(seconds=time)).split('.')[0]} = {time:0>7.2f}sec\n"
        f"Pressure is normalized to [0, 100]. Recent cop: {recent_duration} frames",
        ha="left",
        fontsize=14,
    )
    return fig


def visualize_pressure_data(data, fog, act, id_, trial, fnum, cop=None, lf_cols=slice(0, 7), rf_cols=slice(7, 14)):
    _ = process_pressure_data(data, fog, act, id_, trial, fnum, cop, lf_cols, rf_cols)
    plt.show()


def save_pressure_data_frame(
    data: np.ndarray,
    fogs: np.ndarray,
    acts: np.ndarray,
    id_: int,
    trial: int,
    fnum: int,
    dir: str,
    cop: np.ndarray = None,
    lf_cols=slice(0, 7),
    rf_cols=slice(7, 14),
):
    # data.shape = (neighbors, height, width)
    fig = process_pressure_data(data, fogs, acts, id_, trial, fnum, cop, lf_cols, rf_cols)

    path = os.path.join(dir, f"{id_:02d}", f"{trial:01d}", f"frame_{fnum:05d}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def save_frames(data: np.ndarray, fogs: np.ndarray, acts: np.ndarray, id_: int, trial: int, dir: str, cop: np.ndarray = None):
    num_frames = data.shape[1]
    frame_filenames = []

    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(save_pressure_data_frame, data[:, fnum], fogs, acts, id_, trial, fnum, dir, cop)
            for fnum in range(num_frames)
        ]

        for future in futures:
            frame_filenames.append(future.result())

    return frame_filenames


def calc_cop(data):
    height, width = data.shape
    return np.array([height / 2, width / 2]) if np.all(data == 0) else np.array(ndimage.center_of_mass(data))


if __name__ == "__main__":
    # os.chdir("/workspace")

    loader = PklLoader()
    id_trial_tag = loader.get(type_="tag")
    id_trial_raw = loader.data_id_trial_type

    # for id_ in id_trial_raw.keys():
    #     for trial in id_trial_raw[id_].keys():

    id_, trial = 4, 1
    # Load fog and action class information
    # fog.shape, act.shape = (num_frames,)
    fogs = id_trial_tag[id_][trial]["fog"].to_numpy()
    acts = id_trial_tag[id_][trial]["action"].to_numpy()

    # img.shape = (neighbors, num_frames, height, width)
    img = [raw2img(id_trial_raw[id_][trial][type_].values) for type_ in ["n3", "swing_neigh_r1", "swing_neigh_r2"]]
    img_normed = np.stack(list(map(normalize_data, img)))

    cop = []
    for n in img_normed:
        left_foot_data = n[..., 0:7]
        right_foot_data = n[..., 7:15]

        left_foot_cop = np.stack(list(map(calc_cop, left_foot_data)))
        right_foot_cop = np.stack(list(map(calc_cop, right_foot_data)))

        cop.append(np.concatenate((left_foot_cop, right_foot_cop), axis=-1))

    # cop.shape = (neighbors, num_frames, 4)
    cop = np.stack(cop)

    # Save frames in parallel
    dir = "visualization_results"
    _ = save_frames(data=img_normed, fogs=fogs, acts=acts, id_=id_, trial=trial, dir=dir, cop=cop)
