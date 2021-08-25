from dataclasses import dataclass
from vbfml.input.uproot import UprootReaderMultiFile
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd


@dataclass
class DatasetInfo:
    name: str
    files: list
    n_events: int
    treename: str


class MultiDatasetSequence(Sequence):
    def __init__(self, batch_size: int, branches: "list[str]", shuffle=True) -> None:
        self.datasets = {}
        self.readers = {}
        self.branches = branches
        self.batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self) -> int:
        return self.total_events() // self.batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle

    def _read_dataframes_for_batch(self, idx: int) -> list:
        """Reads and returns data for a given batch and all datasets"""
        dataframes = []
        for name in self.datasets.keys():
            df = self._read_single_dataframe_for_batch_(idx, name)
            dataframes.append(df)
        return dataframes

    def _get_start_stop_for_single_read(self, idx: int, dataset_name: str) -> tuple:
        """Returns the start and stop coordinates for reading a given batch of data from one dataset"""
        start = np.floor(idx * self.batch_size * self.fractions[dataset_name])
        stop = np.floor((idx + 1) * self.batch_size * self.fractions[dataset_name]) - 1
        return start, stop

    def _read_single_dataframe_for_batch_(self, idx: int, dataset_name: str):
        """Reads and returns data for a given batch and single data"""
        start, stop = self._get_start_stop_for_single_read(idx, dataset_name)
        if not dataset_name in self.readers:
            self._initialize_reader(dataset_name)
        df = self.readers[dataset_name].read_events(start, stop)
        df["label"] = dataset_name
        return df

    def __getitem__(self, idx: int) -> tuple:
        """Returns a single batch of data"""
        dataframes = self._read_dataframes_for_batch(idx)

        df = pd.concat(dataframes)

        if self.shuffle:
            df = df.sample(frac=1)

        features = df.drop(columns="label").to_numpy().T
        labels = np.array(df["label"]).reshape((1, len(df["label"])))

        return (features, labels)

    def total_events(self) -> int:
        """Total number of events of all data sets"""
        return sum(dataset.n_events for dataset in self.datasets.values())

    def add_dataset(
        self, name: str, files: "list[str]", n_events: int, treename="tree"
    ) -> None:
        """Add a new data set to the Sequence."""
        if name in self.datasets:
            raise IndexError(f"Dataset already exists: '{name}'.")
        info = DatasetInfo(name=name, files=files, n_events=n_events, treename=treename)
        self.datasets[name] = info
        self._calculate_fractions()

    def _initialize_reader(self, dataset_name) -> None:
        """
        Initializes file readers for a given data set.

        Note that this operation may be slow as the reader
        will open all files associated to it to determine
        event counts.
        """
        info = self.datasets[dataset_name]
        reader = UprootReaderMultiFile(
            files=info.files,
            branches=self.branches,
            treename=info.treename,
            dataset=dataset_name,
        )
        self.readers[dataset_name] = reader

    def _initialize_readers(self) -> None:
        """Initializes file readers for all data sets"""
        for dataset_name in self.datasets.keys():
            self._initialize_reader(dataset_name)

    def _calculate_fractions(self) -> None:
        """Determine what fraction of the total events is from a given data set"""
        total = self.total_events()
        self.fractions = {
            name: info.n_events / total for name, info in self.datasets.items()
        }
