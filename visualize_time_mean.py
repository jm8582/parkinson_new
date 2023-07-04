# time mean
import os

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap, Normalize

import utils
from load_pkl import loader

cmap = LinearSegmentedColormap.from_list("custom", [(0, "black"), (0.00001, "blue"), (0.1, "green"), (1, "red")], N=256)
norm = Normalize(vmin=0, vmax=100)
data = loader.data_id_trial_type


for id_ in data.keys():
    trials = data[id_].keys()
    n_neighs = 4
    fig, axes = plt.subplots(len(trials), n_neighs, figsize=(5 * n_neighs, 5 * len(trials)), constrained_layout=True)
    fig.suptitle(f"ID: {id_}", fontsize=40)
    for trial_i, trial in enumerate(trials):
        d = data[id_][trial]
        ax = axes[trial]
        raws = [(type_, d.get(type_)) for type_ in ["raw", "n1", "n2", "n3"] if d.get(type_) is not None]

        for raws_i, (type_, raw) in enumerate(raws):
            cbar = raws_i == len(raws) - 1

            img = utils.raw2img(raw.values.mean(axis=0))
            sn.heatmap(img, ax=ax[raws_i], xticklabels=False, yticklabels=False, cmap=cmap, norm=norm, cbar=cbar)
            ax[raws_i].axvline(7, color="w")
            if trial_i == 0:
                ax[raws_i].set_title(type_, fontsize=30)
    path = f"visualization_results/time_mean_neighs"
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{id_}.png", facecolor="white")
    plt.close("all")
