import uproot
import pandas as pd
import numpy as np

class SingleDatasetGenerator():
    def __init__(self, files, branches, treename, dataset):
        self.files = files
        self.branches = branches
        self.n_files = len(self.files)
        self.treename = treename
        self.dataset = dataset
        self.file_index = 0
        self.event_index = 0

    def next_file(self):
        self.file_index += 1
        self.event_index = 0

    def reset(self):
        self.file_index = 0
        self.event_index = 0

    def read_events(self, n_events_to_read):
        x, y = None, None
        dataframes = []
        n_events_left_to_read = n_events_to_read

        while n_events_left_to_read:
            # File/tree to read from
            f = uproot.open(self.files[self.file_index])
            tree = f[self.treename]

            # Indices to read between
            # Note that index 'stop' is not read
            # (same logic as for python 'range')
            start = self.event_index
            stop = start + n_events_left_to_read

            # End of file
            # next file in next iteration
            if stop >= tree.num_entries:
                stop = tree.num_entries
                self.next_file()
            else:
                self.event_index = stop + 1

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
            n_events_left_to_read = n_events_left_to_read - n_events_read

        df = pd.concat(dataframes)
        # print(df)
        x = df.to_numpy()
        print(x.shape)
        print((n_events_to_read, len(self.branches)))
        y = np.array([[self.dataset]] * x.shape[0])

        # Sanity
        assert x.shape == (n_events_to_read, len(self.branches))
        assert y.shape == (n_events_to_read, 1)

        return x, y
