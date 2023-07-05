import os
import pickle
import concurrent.futures
from fixed_config import ids, trials


class PklLoader:
    def __init__(self, dset_path="../data/pkl", swing_neigh=True):
        self.dset_path = dset_path
        self.neigh_order = ["tag", "raw", "n1", "n2", "n3"]
        if swing_neigh:
            self.neigh_order += ["swing_neigh_r1", "swing_neigh_r2", "swing_neigh_r3"]

        self.data_id_type_trial = {}
        self.data_id_trial_type = {}
        self.data_type_id_trial = {}
        self.data_type_trial_id = {}
        self.data_trial_id_type = {}
        self.data_trial_type_id = {}

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self.executor.submit(self._init)

    def _init(self):
        self._load_all_data()
        self._cal_data_max_neigh()

    def _load_all_data(self):
        for id_ in ids:
            for type_ in self.neigh_order:
                for trial in trials:
                    self._load_data(id_, type_, trial)

    def _load_data(self, id_, type_, trial):
        if os.path.exists(f"{self.dset_path}/{id_}/{type_}/{trial}.pkl"):
            with open(f"{self.dset_path}/{id_}/{type_}/{trial}.pkl", "rb") as f:
                data = pickle.load(f)

            # Store the data in all dictionaries
            self.data_id_type_trial.setdefault(id_, {}).setdefault(type_, {})[trial] = data
            self.data_id_trial_type.setdefault(id_, {}).setdefault(trial, {})[type_] = data
            self.data_type_id_trial.setdefault(type_, {}).setdefault(id_, {})[trial] = data
            self.data_type_trial_id.setdefault(type_, {}).setdefault(trial, {})[id_] = data
            self.data_trial_id_type.setdefault(trial, {}).setdefault(id_, {})[type_] = data
            self.data_trial_type_id.setdefault(trial, {}).setdefault(type_, {})[id_] = data

    def _cal_data_max_neigh(self):
        self.data_max_neigh = {}
        self.max_neigh = {}
        for id_ in self.data_id_type_trial:
            type_ = max(self.data_id_type_trial[id_].keys(), key=self.neigh_order.index)
            self.data_max_neigh[id_] = self.data_id_type_trial[id_][type_]
            self.max_neigh[id_] = type_

    def get(self, id_=None, type_=None, trial=None):
        self.executor.shutdown(wait=True)

        if id_ is not None and type_ is None and trial is None:
            return self.data_id_type_trial[id_]
        elif id_ is None and type_ is not None and trial is None:
            return self.data_type_id_trial[type_]
        elif id_ is None and type_ is None and trial is not None:
            return self.data_trial_id_type[trial]
        elif id_ is not None and type_ is not None and trial is None:
            return self.data_id_type_trial[id_][type_]
        elif id_ is None and type_ is not None and trial is not None:
            return self.data_type_trial_id[type_][trial]
        elif id_ is not None and type_ is None and trial is not None:
            return self.data_id_trial_type[id_][trial]
        elif id_ is not None and type_ is not None and trial is not None:
            return self.data_id_type_trial[id_][type_][trial]
        else:
            return self.data_id_type_trial
