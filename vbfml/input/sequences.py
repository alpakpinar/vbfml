import uproot
from tensorflow.keras.utils import Sequence

class SingleDatasetSequence(Sequence):
    def __init__(self, files: 'list[str]', branches: 'list[str]', treename:str, dataset:str)->None:
        self.files = files
        self.branches = branches
        self.n_files = len(self.files)
        self.treename = treename
        self.dataset = dataset
        self.nevents_per_file = {}
        self._update_nevents_dict_all_files()

    def _open_file(self, file_index):
        """Current file object to read from."""
        return uproot.open(self._file_path(file_index))

    def _get_tree(self, file_index):
        """Current tree to read from."""
        return self._open_file(file_index)[self.treename]

    def _file_path(self,file_index):
        """Translate file index to file path"""
        return self.files[file_index]
    
    def _update_nevents_dict_single_file(self, file_index):
        """Save number of events in this file into the cache"""
        self.nevents_per_file[file_index] = self._get_tree(file_index).num_entries

    def _update_nevents_dict_all_files(self):
        """Save number of events in all files into the cache"""
        for file_index in range(self.n_files):
            self._update_nevents_dict_single_file(file_index)

    def reset(self):
        pass
