from unittest import TestCase

import numpy as np
import pandas as pd

from vbfml.util import MultiBatchBuffer, DatasetAndLabelConfiguration, vbfml_path


class TestMultiBatchBuffer(TestCase):
    def setUp(self):
        self.batch_size = 3
        self.buffer = MultiBatchBuffer(batch_size=self.batch_size)

        values = {}

        self.n_rows = 10
        values["a"] = np.array(range(self.n_rows))
        values["b"] = np.array(range(self.n_rows, 2 * self.n_rows))
        self.df = pd.DataFrame(values)

    def test_buffer_insertion(self):
        offset = 5
        self.buffer.set_multibatch(self.df, min_batch=offset)

        # Attempt to get non-existing batches
        for index in range(offset):
            self.assertFalse(index in self.buffer)
            with self.assertRaises(IndexError):
                self.buffer.get_batch_df(index)

        for index in range(offset, offset + len(self.df) // self.batch_size):
            self.assertTrue(index in self.buffer)
            df = self.buffer.get_batch_df(index)
            self.assertEqual(len(df.columns), len(self.df.columns))

    def test_batch_values(self):
        self.buffer.set_multibatch(self.df, min_batch=0)
        batch_df_0 = self.buffer.get_batch_df(0)
        self.assertEqual(len(batch_df_0), self.buffer.batch_size)
        self.assertListEqual(list(batch_df_0["a"]), list(range(self.buffer.batch_size)))

        batch_df_1 = self.buffer.get_batch_df(1)
        self.assertEqual(len(batch_df_1), self.buffer.batch_size)
        self.assertListEqual(
            list(batch_df_1["a"]),
            list(range(self.buffer.batch_size, 2 * self.buffer.batch_size)),
        )


class TestDatasetAndLabelConfiguration(TestCase):
    def setUp(self):
        path = vbfml_path("config/datasets/datasets.yml")
        self.dataset_config = DatasetAndLabelConfiguration(path)

    def test_branches(self):
        data = self.dataset_config.data
        regular_exps = []
        # Each branch must have regex and scale parameters
        for label, info in data.items():
            for branch_name in ["regex", "scale"]:
                self.assertIn(branch_name, info)

                if branch_name == "regex":
                    regular_exps.append(info[branch_name])

        # Check if all the regular expressions are unique
        self.assertEqual(len(set(regular_exps)), len(regular_exps))

    def test_n_classes(self):
        labels = self.dataset_config.get_dataset_labels()
        scales = self.dataset_config.get_dataset_scales()
        self.assertEqual(len(labels), len(scales))
