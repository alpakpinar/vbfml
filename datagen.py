from dataclasses import dataclass

import numpy as np
import pandas as pd
import uproot


class SingleDatasetGeneratorUproot():
    """
    Generates data from a series of files.
    """
    def __init__(self, files: 'list[str]', branches: 'list[str]', treename:str, dataset:str)->None:
        self.files = files
        self.branches = branches
        self.n_files = len(self.files)
        self.treename = treename
        self.dataset = dataset
        self.reset()

    def _next_file(self) -> None:
        self.file_index += 1
        self.event_index = 0

    def reset(self) -> None:
        self.file_index = 0
        self.event_index = 0

    def has_events_left(self) -> bool:
        return  self.file_index < len(self.files)

    def _get_file(self) -> str:
        """Current file path to read from."""
        return self.files[self.file_index]

    def _open_file(self):
        """Current file object to read from."""
        return uproot.open(self._get_file())

    def _get_tree(self):
        """Current tree to read from."""
        return self._open_file()[self.treename]

    def read_events(self, n_events_to_read: int) -> np.ndarray:
        """
        Returns a tuple (features, labels) for use in ML.

        """
        dataframes = []
        n_events_left_to_read = n_events_to_read
        while n_events_left_to_read:
            if not self.has_events_left():
                raise EOFError("No more events to read.")

            # Tree to read from
            tree = self._get_tree()

            # Indices to read between
            # Note that index 'stop' is not read
            # (same logic as for python 'range')
            start = self.event_index
            stop = start + n_events_left_to_read
            # End of file
            # next file in next iteration
            if stop > tree.num_entries:
                stop = tree.num_entries
                self._next_file()
            else:
                self.event_index = stop

            # Read
            df = tree.arrays(
                expressions=self.branches,
                entry_start=start,
                entry_stop=stop,
                library='pandas'
            )

            dataframes.append(df)

            # How much more do we need to read
            n_events_read = stop - start
            assert(len(df) == n_events_read)
            n_events_left_to_read = n_events_left_to_read - n_events_read

        df = pd.concat(dataframes)
        # print(df)
        x = df.to_numpy()
        y = np.array([[self.dataset]] * x.shape[0])

        # Sanity
        assert x.shape == (n_events_to_read, len(self.branches))
        assert y.shape == (n_events_to_read, 1)

        return x, y

@dataclass
class DatasetInfo():
    name: str
    xs: float
    files: list
    nevents: int

class MultiDatasetGenerator():
    def __init__(self):
        self.datasets = {}
        self.generators = {}
        self.total_events = 0
        self.batch_size = 100
        self.fractions = {}

    def add_dataset(self, dataset, kwargs):
        self.datasets[dataset.name] = dataset
        self.generators[dataset.name] = SingleDatasetGenerator(dataset.files, dataset=dataset, **kwargs)
        self.total_events += dataset.nevents


    def build(self):
        self.n_batches = self.total_events // self.batch_size
        for name, dataset in self.datasets.items():
            self.fractions[name] = dataset.nevents / self.total_events

    def __getitem__(self, index):
        if index > self.n_batches:
            raise IndexError
