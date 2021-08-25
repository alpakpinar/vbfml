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


class MultiDatasetSequence(Sequence):
    def __init__(self, batch_size: int, branches: "list[str]", shuffle=True) -> None:
        self.datasets = {}
        self.readers = {}
        self.branches = branches
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.total_events // self.batch_size

    def __get_item__(self, idx: int) -> tuple:
        dataframes = []
        for name, _ in self.datasets.keys():
            start = idx * self.batch_size * self.fractions[name]
            stop = (idx + 1) * self.batch_size * self.fractions[name]
            df = self.readers[name].read_events(start, stop)
            df["label"] = name
            dataframes.append(df)

        if self.shuffle:
            df = pd.concat(dataframes)
        df = df.sample(frac=1)

        features = df.drop("label").to_numpy().T
        labels = df["label"]

        return (features, labels)

    def total_events(self) -> int:
        return np.sum(dataset.n_events for dataset in self.datasets.values())

    def add_dataset(self, name: str, files: "list[str]", n_events: int) -> None:
        info = DatasetInfo(name=name, files=files, n_events=n_events)
        self.datasets[name] = info

    def _initialize_readers(self) -> None:
        for name, info in self.datasets.items():
            reader = UprootReaderMultiFile(
                files=info.files, branches=self.branches, treename=info.treename
            )
            self.readers[name] = reader

    def _calculate_fractions(self) -> None:
        total = self.total_events()
        self.fractions = {
            name: info.n_events / total for name, info in self.datasets.items()
        }
