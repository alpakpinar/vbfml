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

    def _index_into_file(self, global_event_index):
        """
        Translate global file index to tuple of (file index, local event index).
        
        A global event index is just the index of the desired events in the list of all events in all files (between 0 and total event number)
        
        The file index is the index of the file that contains this event, and the local event index is the index of the given event counting from the beginning of the file.
        """
        events_before = 0

        target_file_index = None
        local_event_index = None

        for file_index in range(self.n_files):
            nevents = self.nevents_per_file[file_index]
        
            right_file =  global_event_index < events_before + nevents
            if right_file:
                target_file_index = file_index
                local_event_index = global_event_index - events_before
                break
            events_before += nevents

        return (target_file_index, local_event_index)

    def read_events(self, start, stop):
        return None, None

    def reset(self):
        pass
