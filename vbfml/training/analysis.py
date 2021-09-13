import os
import pickle
from dataclasses import dataclass, field

import hist
import numpy as np
from tqdm import tqdm

from vbfml.input.sequences import MultiDatasetSequence
from vbfml.training.data import TrainingLoader

# Hard coded sane default binnings
# If it extends more, might warrant
# separate configuration file.
dphi_bins = np.linspace(0, np.pi, 20)
phi_bins = np.linspace(-np.pi, np.pi, 20)
pt_bins = np.logspace(2, 3, 20)
eta_bins = np.linspace(-5, 5, 20)
axis_bins = {
    "mjj": np.logspace(2, 4, 20),
    "dphijj": dphi_bins,
    "detajj": np.linspace(0, 10, 20),
    "recoil_pt": pt_bins,
    "dphi_ak40_met": dphi_bins,
    "dphi_ak41_met": dphi_bins,
    "ht": pt_bins,
    "leadak4_pt": pt_bins,
    "leadak4_phi": phi_bins,
    "leadak4_eta": eta_bins,
    "trailak4_pt": pt_bins,
    "trailak4_phi": phi_bins,
    "trailak4_eta": eta_bins,
    "score": np.linspace(0, 1, 20),
}


@dataclass
class TrainingAnalyzer:
    """
    Analyzer to make histograms based on
    training / validation / test data sets.

    """

    directory: str
    cache: str = "analyzer_cache.pkl"
    histograms: "dict" = field(default_factory=dict)

    def __post_init__(self):
        self.loader = TrainingLoader(self.directory)
        self.cache = os.path.join(self.directory, os.path.basename(self.cache))

    def _make_histogram(self, quantity_name: str) -> hist.Hist:
        """Creates an empty histogram for a given quantity (feature or score)"""
        if "score" in quantity_name:
            bins = axis_bins["score"]
        else:
            bins = axis_bins[quantity_name]
        histogram = hist.Hist(
            hist.axis.Variable(bins, name=quantity_name, label=quantity_name),
            hist.axis.IntCategory(range(10), name="label", label="label"),
            storage=hist.storage.Weight(),
        )

        return histogram

    def load_from_cache(self):
        success = False
        try:
            with open(self.cache, "rb") as f:
                histograms = pickle.load(f)
            success = True
        except:
            histograms = {}
        self.histograms = histograms
        return success

    def write_to_cache(self):
        with open(self.cache, "wb") as f:
            return pickle.dump(self.histograms, f)

    def analyze(self):
        """
        Loads all relevant data sets and analyze them.
        """
        histograms = {}
        for sequence_type in ["training", "validation"]:
            sequence = self.loader.get_sequence(sequence_type)
            sequence.scale_features = False
            sequence.batch_size = int(1e6)
            sequence.batch_buffer_size = 10
            histograms[sequence_type] = self._analyze_sequence(sequence, sequence_type)
        self.histograms = histograms

    def _analyze_sequence(
        self, sequence: MultiDatasetSequence, sequence_type: str
    ) -> "dict[str:hist.Hist]":
        """
        Analyzes a specific sequence.

        Loop over batches, make and return histograms.
        """
        histograms = {}
        model = self.loader.get_model()
        feature_names = self.loader.get_features()

        feature_scaler = self.loader.get_feature_scaler()

        for ibatch in tqdm(
            range(len(sequence)), desc=f"Analyze batches of {sequence_type} sequence."
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)

            scores = model.predict(feature_scaler.transform(features))

            # Histogramming of features
            for ifeat, feature_name in enumerate(feature_names):
                if not feature_name in histograms:
                    histograms[feature_name] = self._make_histogram(feature_name)
                histograms[feature_name].fill(
                    **{
                        feature_name: features[:, ifeat].flatten(),
                        "label": labels,
                        "weight": weights.flatten(),
                    }
                )

            # Histogramming of NN scores
            n_classes = labels_onehot.shape[1]
            for scored_class in range(n_classes):
                name = f"score_{scored_class}"
                if not name in histograms:
                    histograms[name] = self._make_histogram(name)
                histograms[name].fill(
                    **{
                        name: scores[:, scored_class],
                        "label": labels,
                        "weight": weights,
                    }
                )
        return histograms
