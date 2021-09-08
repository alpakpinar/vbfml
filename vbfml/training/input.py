import re
import os
import glob
from vbfml.input.sequences import MultiDatasetSequence, DatasetInfo
from vbfml.training.util import get_n_events
from tabulate import tabulate


def summarize_datasets(datasets: "list[DatasetInfo]") -> None:
    """
    Prints a neat summary of a group of datasets to the terminal.
    """
    table = []
    for dataset in datasets:
        table.append((dataset.label, dataset.name, dataset.n_events))
    print(
        tabulate(
            sorted(table),
            headers=["Class label", "Physics data set name", "Number of events"],
        )
    )


def build_sequence(
    datasets: "list[DatasetInfo]", features: "list[str]", absolute_weight=False
) -> MultiDatasetSequence:
    """Shortcut to set up a MultiDatasetSequence"""

    weight_expression = "weight_total * xs / sumw"
    if absolute_weight:
        weight_expression = f"abs({weight_expression})"
    sequence = MultiDatasetSequence(
        batch_size=50,
        branches=features,
        shuffle=True,
        batch_buffer_size=int(1e5),
        weight_expression="weight_total*xs/sumw",
    )

    for dataset in datasets:
        sequence.add_dataset(dataset)

    return sequence


def dataset_from_file_name_bucoffea(filepath: str) -> str:
    """
    Decodes data set name from file name, assuming bucoffea origin.
    """
    base = os.path.basename(filepath)
    m = re.match("tree_(.*_\d{4}).root", base)
    dataset_name = m.groups()[0]
    return dataset_name


def load_datasets_bucoffea(directory: str) -> "list[DatasetInfo]":
    """
    Load data sets based on a directory of ROOT files from bucoffea.

    Data set names are decoded from the file names, assuming one file per data set.
    By default, the labels and names of the data sets are set equal.
    """
    files = glob.glob(f"{directory}/*root")
    datasets = []
    for file in files:
        dataset_name = dataset_from_file_name_bucoffea(file)
        if "2018" in dataset_name:
            continue

        n_events = get_n_events(file, "sr_vbf")
        if not n_events:
            continue

        dataset = DatasetInfo(
            name=dataset_name,
            label=dataset_name,
            files=[file],
            treename="sr_vbf",
            n_events=int(n_events),
            weight=1,
        )
        datasets.append(dataset)
    return datasets


def select_and_label_datasets(
    all_datasets: "list[DatasetInfo]", labels: "dict[str:str]"
) -> "list[DatasetInfo]":
    """
    Slim down a list of datasets and apply labeling.

    Labels is a dictionary that maps a new target label name to a regular
    expression matching data set names. Only data sets matching one of the
    regular expressions are returned. The labels of these data sets are
    set to the corresponding key from the labels dict.
    """
    selected_datasets = []
    for label, regex in labels.items():
        matching_datasets = [
            dataset for dataset in all_datasets if re.match(regex, dataset.name)
        ]
        for dataset in matching_datasets:
            dataset.label = label
        selected_datasets.extend(matching_datasets)
    return selected_datasets
