import os
import pickle
import keras
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

import hist
import numpy as np
import sklearn
from tqdm import tqdm
from vbfml.input.sequences import MultiDatasetSequence
from vbfml.training.data import TrainingLoader

pjoin = os.path.join

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
    "mjj_maxmjj": np.logspace(2, 4, 20),
    "dphijj_maxmjj": dphi_bins,
    "detajj_maxmjj": np.linspace(0, 10, 20),
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
    "leadak4_mjjmax_pt": pt_bins,
    "leadak4_mjjmax_phi": phi_bins,
    "leadak4_mjjmax_eta": eta_bins,
    "trailak4_mjjmax_pt": pt_bins,
    "trailak4_mjjmax_phi": phi_bins,
    "trailak4_mjjmax_eta": eta_bins,
    "score": np.linspace(0, 1, 51),
    "composition": np.linspace(-0.5, 4.5, 5),
    "transformed": np.linspace(-5, 5, 20),
}


@dataclass
class TrainingAnalyzerBase:
    """
    Base class for training analyzers.
    Contains methods that are common to all analyzers.
    """

    directory: str
    cache: str = "analyzer_cache.pkl"

    data: "dict" = field(default_factory=dict)

    def __post_init__(self):
        self.loader = TrainingLoader(self.directory,framework="pytorch")
        self.cache = os.path.join(self.directory, os.path.basename(self.cache))

    def _make_histogram(self, quantity_name: str) -> hist.Hist:
        """Creates an empty histogram for a given quantity (feature or score)"""
        if "score" in quantity_name:
            bins = axis_bins["score"]
        elif "transform" in quantity_name:
            bins = axis_bins["transformed"]
        else:
            bins = axis_bins[quantity_name]
        histogram = hist.Hist(
            hist.axis.Variable(bins, name=quantity_name, label=quantity_name),
            hist.axis.IntCategory(range(10), name="label", label="label"),
            storage=hist.storage.Weight(),
        )

        return histogram

    def _fill_score_histograms(
        self,
        histograms: Dict[str, hist.Hist],
        scores: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Create and fill score histograms.

        Args:
            histograms (Dict[str, hist.Hist]): The dictionary of histogram objects.
            scores (np.ndarray): Array of predicted scores.
            labels (np.ndarray): Array of truth labels.
            weights (np.ndarray): Array of sample weights.
        """
        n_classes = scores.shape[1]
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

    def load_from_cache(self) -> bool:
        """Histograms loaded from disk cache."""
        success = False
        try:
            with open(self.cache, "rb") as f:
                data = pickle.load(f)
            success = True
        except:
            data = {}
        self.data = data
        return success

    def write_to_cache(self) -> bool:
        """Cached histograms written to disk."""
        with open(self.cache, "wb") as f:
            return pickle.dump(self.data, f)


@dataclass
class TrainingAnalyzer(TrainingAnalyzerBase):
    """
    Analyzer to make histograms based on
    training / validation / test data sets.

    """

    def analyze(self):
        """
        Loads all relevant data sets and analyze them.
        """
        histograms = {}
        for sequence_type in ["training", "validation"]:
            sequence = self.loader.get_sequence(sequence_type)
            sequence.scale_features = "none"
            sequence.batch_size = int(1e6)
            sequence.batch_buffer_size = 10
            (
                histogram_out,
                predicted_scores,
                validation_scores,
                weights,
            ) = self._analyze_sequence(sequence, sequence_type)
            histograms[sequence_type] = histogram_out
            if sequence_type == "validation":
                self.data["truth_scores"] = validation_scores
                self.data["predicted_scores"] = predicted_scores
                self.data["weights"] = weights
        self.data["histograms"] = histograms

    def _fill_feature_histograms(
        self,
        histograms: Dict[str, hist.Hist],
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        feature_scaler: Optional[sklearn.preprocessing.StandardScaler],
    ):
        feature_names = self.loader.get_features()

        if feature_scaler:
            features = feature_scaler.transform(features)

        for ifeat, feature_name in enumerate(feature_names):
            if feature_scaler:
                feature_name = f"{feature_name}_transform"
            if not feature_name in histograms:
                histograms[feature_name] = self._make_histogram(feature_name)
            histograms[feature_name].fill(
                **{
                    feature_name: features[:, ifeat].flatten(),
                    "label": labels,
                    "weight": weights.flatten(),
                }
            )

    def _fill_feature_covariance(
        self, features: np.ndarray, labels: np.ndarray, feature_scaler
    ):
        """
        Calculate covariance coefficients between different features (e.g. mjj vs met).

        Covariance coefficients are stored in a dictionary member of the analyzer.
        For each batch of feature values, the previously calculated covariance is updated
        by calculating the weighted mean between the old and new covariance values,
        using the underlying number of events as the weight for the mean.

        To make this procedure more accurate, the feature covariance is calculated
        after feature scaling, which ensures that the individual feature averages
        are close to zero, which simplifies the formula for the covariance.
        """
        feature_names = self.loader.get_features()
        features = feature_scaler.transform(features)

        if not "covariance" in self.data:
            self.data["covariance"] = defaultdict(dict)
            self.data["covariance_event_count"] = defaultdict(int)

        for iclass in range(labels.shape[1]):
            n_event_previous = self.data["covariance_event_count"][iclass]
            mask = labels[:, iclass] == 1
            n_events = np.sum(mask)
            for ifeat1, feat1 in enumerate(feature_names):
                for ifeat2, feat2 in enumerate(feature_names):
                    if ifeat2 < ifeat1:
                        continue

                    covariance = np.cov(
                        features[mask][:, ifeat1], features[mask][:, ifeat2]
                    )

                    key = tuple(sorted((feat1, feat2)))

                    previous_covariance = self.data["covariance"][iclass].get(key, 0)
                    updated_covariance = (
                        previous_covariance * n_event_previous + covariance * n_events
                    )
                    updated_covariance /= n_event_previous + n_events
                    self.data["covariance"][iclass][key] = updated_covariance

            self.data["covariance_event_count"][iclass] = n_event_previous + n_events

    def _analyze_sequence(
        self, sequence: MultiDatasetSequence, sequence_type: str
    ) -> Dict[str, hist.Hist]:
        """
        Analyzes a specific sequence.

        Loop over batches, make and return histograms.
        """
        histograms = {}
        model = self.loader.get_model()

        feature_scaler = self.loader.get_feature_scaler()
        predicted_scores = []
        validation_scores = []
        sample_weights = []
        for ibatch in tqdm(
            range(len(sequence)), desc=f"jk Analyze batches of {sequence_type} sequence."
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)

            scores = model.predict(feature_scaler.transform(features))
            print(scores)
            if sequence_type == "validation":
                predicted_scores.append(scores)
                validation_scores.append(labels_onehot)
                sample_weights.append(weights)

            for scaler in feature_scaler, None:
                self._fill_feature_histograms(
                    histograms, features, labels, weights, feature_scaler=scaler
                )

            self._fill_score_histograms(histograms, scores, labels, weights)
            self._fill_feature_covariance(features, labels_onehot, feature_scaler)
            # self._fill_composition_histograms(histograms, scores, labels, weights)

        return histograms, predicted_scores, validation_scores, weights


@dataclass
class ImageTrainingAnalyzer(TrainingAnalyzerBase):
    def _group_images(
        self,
        features: np.ndarray,
        predicted_scores: np.ndarray,
        truth_labels: np.ndarray,
        weights: np.ndarray,
    ):
        """
        Groups the list of images into "correctly classified" and "mis-classified".
        Returns a dictionary containing the list of images for both.
        """
        predicted_labels = predicted_scores.argmax(axis=1)
        wrong_clf = predicted_labels != truth_labels

        # Return data grouped into a few classes:
        # Mis-classified images and correctly classified images
        grouping = {}
        grouping["mis_classified"] = {
            "features": features[wrong_clf],
            "scores": predicted_scores[wrong_clf],
            "truth_labels": truth_labels[wrong_clf],
            "weights": weights[wrong_clf],
        }

        grouping["correctly_classified"] = {
            "features": features[~wrong_clf],
            "scores": predicted_scores[~wrong_clf],
            "truth_labels": truth_labels[~wrong_clf],
            "weights": weights[~wrong_clf],
        }

        # Also take a look at the images where the model strongly predicts wrongly
        n_classes = predicted_scores.shape[1]

        # Strongly mis-classified samples
        for i in range(n_classes):
            scores = predicted_scores[:, i]
            tag = f"truth_{i}_strongly_mis_clf"
            mask = (scores < 0.1) & (truth_labels == i)

            grouping[tag] = {
                "features": features[mask],
                "scores": predicted_scores[mask],
                "truth_labels": truth_labels[mask],
                "weights": weights[mask],
            }

        # Strongly well-classified samples
        for i in range(n_classes):
            scores = predicted_scores[:, i]
            tag = f"truth_{i}_strongly_clf"
            mask = (scores > 0.9) & (truth_labels == i)

            grouping[tag] = {
                "features": features[mask],
                "scores": predicted_scores[mask],
                "truth_labels": truth_labels[mask],
                "weights": weights[mask],
            }

        return grouping

    def _analyze_sequence(self, sequence: MultiDatasetSequence, sequence_type: str):
        """
        Analyzes a specific sequence.
        """
        histograms = {}
        model = self.loader.get_model()

        predicted_scores = []
        truth_scores = []
        sample_weights = []

        # Obtain the label encoding for this sequence
        label_encoding = {}
        for key, label in sequence.label_encoding.items():
            if not isinstance(key, int):
                continue
            label_encoding[key] = label

        for ibatch in tqdm(
            range(len(sequence)), desc=f"Analyze batches of {sequence_type} sequence"
        ):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)

            # Count the occurence of number of classes in the training and validation sequences
            counter = Counter()
            # Count the instances from each class while taking weights into account
            for label, weight in zip(labels, weights):
                counter.update({label: weight})

            scores = model.predict(features)
            sample_weights.append(weights)

            predicted_scores.append(scores)
            truth_scores.append(labels_onehot)
            

            self._fill_score_histograms(histograms, scores, labels, weights)

        def pretty_labels(index):
            """
            0 -> EWK V+jets/VBF H(inv),
            1 -> QCD V+jets
            """
            if index == 0:
                return "EWK V/VBF H"
            return "QCD V"

        # Normalize the counter values + add the pretty labels
        sample_counts = OrderedDict()
        total_count = sum(counter.values())

        for key, count in counter.items():
            sample_counts[pretty_labels(key)] = (count / total_count)[0]

        # Sort by key
        sample_counts = sorted(sample_counts.items())

        sample_weights = np.vstack(sample_weights).flatten()
        predicted_scores = np.vstack(predicted_scores)
        truth_scores = np.vstack(truth_scores)
        return (
            histograms,
            sample_counts,
            label_encoding,
            predicted_scores,
            truth_scores,
            sample_weights,
        )

    def analyze(self, sequence_types: List[str]):
        """
        Loads all relevant data sets and analyze them.
        """
        histograms = {}
        sample_weights = {}
        sample_counts_per_sequence = {}

        truth_scores = {}
        predicted_scores = {}

        # We'll analyze the training and validation sequences and save histograms
        # for each sequence type.
        # For images, we're typically interested in:
        # - Features (i.e. list of 40x20 images)
        # - List of prediction scores
        # - List of labels
        # - Score distributions for a given class

        for sequence_type in sequence_types:
            sequence = self.loader.get_sequence(sequence_type)
            # sequence.scale_features = "norm"
            sequence.batch_size = int(1e6)
            sequence.batch_buffer_size = 10
            
            (
                histogram_out,
                sample_counts,
                label_encoding,
                _predicted_scores,
                _truth_scores,
                weights,
            ) = self._analyze_sequence(sequence, sequence_type)

            histograms[sequence_type] = histogram_out
            sample_counts_per_sequence[sequence_type] = sample_counts
            sample_weights[sequence_type] = weights

            truth_scores[sequence_type] = _truth_scores
            predicted_scores[sequence_type] = _predicted_scores

        self.data["weights"] = sample_weights
        self.data["histograms"] = histograms
        self.data["sample_counts_per_sequence"] = sample_counts_per_sequence
        self.data["label_encoding"] = label_encoding

        self.data["truth_scores"] = truth_scores
        self.data["predicted_scores"] = predicted_scores
