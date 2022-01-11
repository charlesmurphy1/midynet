import h5py
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
from collections import defaultdict

# from midynet.util import Verbose, MCStatistics


class Metrics:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.names = []
        self.get_data = {}
        self.num_updates = 0
        self.num_procs = config.get("num_procs", 1)
        self.statistics = MCStatistics(self.config.get("error_type", "confidence"))


#     def initialize(self, experiment):
#         self.num_procs = experiment.num_procs
#         self.labels_to_scan = self.config_array.labels_to_scan
#         self.num_updates = len(self.config_array)
#
#     def exit(self, experiment):
#         return
#
#     def eval(self, config):
#         raise NotImplementedError()
#
#     def compute(self, experiment, verbose=Verbose()):
#         self.verbose = verbose
#         self.initialize(experiment)
#
#         pb = self.verbose.init_progress(
#             self.__class__.__name__, total=len(self.config_array)
#         )
#         raw_data = defaultdict(list)
#         for c in self.config_array:
#             val = self.eval(c)
#             for k, v in val.items():
#                 raw_data[k].append(v)
#             if pb is not None:
#                 pb.update()
#             self.verbose.update_progress()
#         self.verbose.end_progress()
#
#         self.data = self.format_data(raw_data)
#         self.exit(experiment)
#
#     def format_data(self, data):
#         formatted_data = {}
#         for k, v in data.items():
#             formatted_data[k] = np.zeros(
#                 [len(self.config_array.config[l]) for l in self.labels_to_scan]
#             )
#             for i, c in enumerate(self.config_array):
#                 index = [
#                     np.where(c.scan[l] == np.array(self.config_array.config[l]))[0]
#                     for j, l in enumerate(self.labels_to_scan)
#                 ]
#                 formatted_data[k][tuple(index)] = v[i]
#         return formatted_data
#
#     def unformat_data(self, data):
#         unformatted_data = {}
#         for k, v in data.items():
#             if v.shape != (
#                 len(self.config_array.config[l]) for l in self.labels_to_scan
#             ):
#                 unformatted_data[k] = v
#             else:
#                 unformatted_data[k] = np.zeros(len(self.config_array))
#                 for i, c in enumerate(self.config_array):
#                     index = [
#                         np.where(c.scan[l] == np.array(self.config_array.config[l]))[0]
#                         for j, l in enumerate(self.labels_to_scan)
#                     ]
#                     unformatted_data[k][i] = v[tuple(index)]
#         return unformatted_data
#
#     def update(self, data):
#         self.data.update(data)
#
#     def save(self, h5file, name=None):
#         if not isinstance(h5file, (h5py.File, h5py.Group)):
#             raise ValueError("Dataset file format must be HDF5.")
#
#         name = name or self.__class__.__name__
#
#         for k, v in self.data.items():
#             path = name + "/" + str(k)
#             if path in h5file:
#                 del h5file[path]
#             h5file.create_dataset(path, data=v)
#
#     def load(self, h5file, name=None):
#         if not isinstance(h5file, (h5py.File, h5py.Group)):
#             raise ValueError("Dataset file format must be HDF5.")
#
#         name = name or self.__class__.__name__
#
#         if name in h5file:
#             self.data = self.read_h5_recursively(h5file[name])
#
#     def read_h5_recursively(self, h5file, prefix=""):
#         ans_dict = {}
#         for key in h5file:
#             item = h5file[key]
#             if prefix == "":
#                 path = f"{key}"
#             else:
#                 path = f"{prefix}/{key}"
#
#             if isinstance(item, h5py.Dataset):
#                 ans_dict[path] = item[...]
#             elif isinstance(item, h5py.Group):
#                 d = self.read_h5_recursively(item, path)
#                 ans_dict.update(d)
#             else:
#                 raise ValueError()
#         return ans_dict
#
#     def estimate_running_time(self):
#         raise NotImplementedError()
#
#
# class CustomMetrics(Metrics):
#     def initialize(self, experiment):
#         return

if __name__ == "__main__":
    pass
