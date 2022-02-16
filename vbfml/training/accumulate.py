import os
import pickle
import numpy as np
import matplotlib.colors

from typing import List, Dict
from dataclasses import dataclass, field
from matplotlib import pyplot as plt

from tqdm import tqdm
from vbfml.input.sequences import MultiDatasetSequence
from vbfml.training.data import TrainingLoader

pjoin = os.path.join


@dataclass
class ImageAccumulator:
    """
    Class to accumulate images and produce a new image
    based on the average of all input images.
    """

    directory: str
    image_shape: tuple = (40, 20)

    avg_images: "dict" = field(default_factory=dict)

    def __post_init__(self):
        self.loader = TrainingLoader(self.directory)

    def _analyze_sequence(
        self, sequence: MultiDatasetSequence, groupby: str, label_encoding: Dict[int, str]
    ) -> Dict[str, np.ndarray]:
        """
        Analyzes a specific sequence.
        """
        imagedict = {}
        weightdict = {}
        model = self.loader.get_model()
        feature_scaler = self.loader.get_feature_scaler()

        for ibatch in tqdm(range(len(sequence)), desc="Accumulating image data"):
            features, labels_onehot, weights = sequence[ibatch]
            labels = labels_onehot.argmax(axis=1)
            weights = weights.flatten()

            scores = model.predict(features)
            predicted_labels = scores.argmax(axis=1)

            # Now, we'll groupby the labels (predicted or truth)
            # of the images, and accumulate them
            if groupby == "truth_label":
                imagedict[label_encoding[0]], weightdict[label_encoding[0]] = (
                    features[labels == 0],
                    weights[labels == 0],
                )
                imagedict[label_encoding[1]], weightdict[label_encoding[1]] = (
                    features[labels == 1],
                    weights[labels == 1],
                )
            elif groupby == "predicted_label":
                imagedict[label_encoding[0]], weightdict[label_encoding[0]] = (
                    features[predicted_labels == 0],
                    weights[predicted_labels == 0],
                )
                imagedict[label_encoding[1]], weightdict[label_encoding[1]] = (
                    features[predicted_labels == 1],
                    weights[predicted_labels == 1],
                )
            else:
                raise ValueError(
                    'groupby can be either: "truth_label" or "predicted_label"'
                )

            # Compute the mean per class
            avg_images = {}
            for imclass, imagelist in imagedict.items():
                avg_images[imclass] = np.average(
                    imagelist, axis=0, weights=weightdict[imclass]
                )

        return avg_images

    def accumulate(self, groupby: str, sequence_type: str = "validation"):
        """
        Function to call to do the accumulation operation.
        """
        sequence = self.loader.get_sequence(sequence_type)
        sequence.scale_features = "norm"
        sequence.batch_size = 10000
        sequence.batch_buffer_size = 1

        # Retrieve label encoding for this sequence
        label_encoding = {}
        for key, label in sequence.label_encoding.items():
            if isinstance(key, str):
                continue
            label_encoding[key] = label

        self.avg_images = self._analyze_sequence(sequence, groupby=groupby, label_encoding=label_encoding)

    def plot(self, groupby: str):
        """
        Function to plot the averaged images.
        """
        etabins = np.linspace(-5, 5, self.image_shape[0])
        phibins = np.linspace(-np.pi, np.pi, self.image_shape[1])

        for imclass, im in tqdm(
            self.avg_images.items(), desc="Plotting averaged images"
        ):
            fig, ax = plt.subplots()
            im = np.reshape(im, self.image_shape)
            cmap = ax.pcolormesh(
                etabins,
                phibins,
                im.T,
                norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e-2),
            )

            cb = fig.colorbar(cmap)
            cb.set_label("Averaged Weighted Energy per Pixel (GeV)")

            ax.text(
                1,
                1,
                imclass,
                fontsize=13,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )
            ax.text(
                0,
                1,
                f"Grouped by: {groupby}",
                fontsize=13,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

            ax.set_xlabel(r"PF Candidate $\eta$")
            ax.set_ylabel(r"PF Candidate $\phi$")

            outdir = pjoin(self.directory, "plots", "accumulated")
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            outpath = pjoin(outdir, f"images_groupby_{groupby}_{imclass}.pdf")
            fig.savefig(outpath)
            plt.close(fig)
