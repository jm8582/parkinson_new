import os
from matplotlib.colors import LinearSegmentedColormap, Normalize

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

import utils
from find_time_avg_at_valley import find_time_avg_at_valley
from load_pkl import loader

# swing phase error neighs
data = loader.data_id_trial_type

for id_ in data.keys():
    trials = data[id_].keys()
    n_neighs = 6
    # n_neighs = len(data[id_][0]) - 1
    fig, axes = plt.subplots(len(trials), n_neighs, figsize=(5 * n_neighs, 5 * len(trials)), constrained_layout=True)
    fig.suptitle(f"ID: {id_}", fontsize=40)
    for trial_i, trial in enumerate(trials):
        d = data[id_][trial]
        ax = axes[trial]
        raws = [
            (type_, d.get(type_))
            for type_ in ["raw", "n1", "n2", "n3", "swing_neigh_r1", "swing_neigh_r2"]
            if d.get(type_) is not None
        ]
        tag = d.get("tag")

        for raws_i, (type_, raw) in enumerate(raws):
            cbar = raws_i == len(raws) - 1

            l_time_avg_at_valley, r_time_avg_at_valley = find_time_avg_at_valley(raw, tag)
            img = utils.raw2img(np.concatenate([l_time_avg_at_valley, r_time_avg_at_valley]))

            cmap = LinearSegmentedColormap.from_list(
                "custom", [(0, "black"), (0.00001, "blue"), (0.1, "green"), (1, "red")], N=256
            )
            norm = Normalize(vmin=0, vmax=100)
            sn.heatmap(img, ax=ax[raws_i], xticklabels=False, yticklabels=False, cmap=cmap, norm=norm, cbar=cbar)
            ax[raws_i].axvline(7, color="w")
            if trial_i == 0:
                ax[raws_i].set_title(type_, fontsize=30)
    path = f"visualization_results/swing_phase_error_neighs"
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{id_}.png", facecolor="white")
    plt.close("all")
