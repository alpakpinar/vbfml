import copy
import os
from datetime import datetime

from vbfml.models import sequential_dense_model
from vbfml.training.data import save
from vbfml.training.input import (
    build_sequence,
    load_datasets_bucoffea,
    select_and_label_datasets,
)
from vbfml.training.util import normalize_classes

features = [
    "mjj",
    "dphijj",
    "detajj",
    "recoil_pt",
    "dphi_ak40_met",
    "dphi_ak41_met",
    "ht",
    "leadak4_pt",
    "leadak4_phi",
    "leadak4_eta",
    "trailak4_pt",
    "trailak4_phi",
    "trailak4_eta",
]
all_datasets = load_datasets_bucoffea(
    directory="/data/cms/vbfml/2021-08-25_treesForML/"
)

dataset_labels = {
    "ewk_17": "EWK.*2017",
    "v_qcd_nlo_17": "(WJetsToLNu_Pt-\d+To.*|Z\dJetsToNuNu_M-50_LHEFilterPtZ-\d+To\d+)_MatchEWPDG20-amcatnloFXFX_2017)",
    "signal_17": "VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017",
}
datasets = select_and_label_datasets(all_datasets, dataset_labels)
normalize_classes(datasets)
training_sequence = build_sequence(dataset=datasets, features=features)


# Training sequence = 90% of total
training_sequence.read_range = (0.0, 0.9)
training_sequence.scale_features = True
training_sequence[0]

# Validation sequence = 10% of total
validation_sequence = copy.deepcopy(training_sequence)
validation_sequence.read_range = (0.9, 1.0)

# Build model
model = sequential_dense_model(
    n_layers=6,
    n_nodes=[13, 13, 8, 8, 4, 4],
    n_features=len(features),
    n_classes=len(training_sequence.dataset_labels()),
)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


steps_total = len(training_sequence)
steps_per_epoch = 50000
epochs = 5 * steps_total // steps_per_epoch

model.fit(
    x=training_sequence,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    max_queue_size=0,
    shuffle=False,
    validation_data=validation_sequence,
)


# Save all kinds of output
name = datetime.now().strftime("%Y-%m-%d_%H-%M")
training_directory = os.path.join("./output", f"model_{name}")

# The trained model
model.save(training_directory)


def prepend_path(fname):
    return os.path.join(training_directory, fname)


# Training history
save(model.history.history, prepend_path("history.pkl"))

# Feature scaling object for future evaluation
save(training_sequence._feature_scaler, prepend_path("feature_scaler.pkl"))

# List of features
save(
    features,
    prepend_path(
        "features.pkl",
    ),
)

# Training and validation sequences
save(training_sequence, prepend_path("training_sequence.pkl"))
save(validation_sequence, prepend_path("validation_sequence.pkl"))
