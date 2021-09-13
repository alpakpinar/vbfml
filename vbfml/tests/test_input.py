from unittest import TestCase

from vbfml.training.input import (
    dataset_from_file_name_bucoffea,
)


class TestInput(TestCase):
    def test_dataset_from_file_name_bucoffea(self):
        validation_pairs = []
        for process in ["zjet", "wjet", "akajsdbk"]:
            for year in [2016, 2017, 2018, 9999]:
                dataset = f"{process}_{year}"
                file_name = f"tree_{dataset}.root"
                validation_pairs.append((file_name, dataset))

        for input_value, expected_output_value in validation_pairs:
            self.assertEqual(
                dataset_from_file_name_bucoffea(input_value), expected_output_value
            )
            self.assertEqual(
                dataset_from_file_name_bucoffea("/a/b/c/" + input_value),
                expected_output_value,
            )
            self.assertEqual(
                dataset_from_file_name_bucoffea("/" + input_value),
                expected_output_value,
            )
