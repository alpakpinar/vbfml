import os
from unittest import TestCase

from vbfml.models import sequential_dense_model
from vbfml.tests.util import make_tmp_dir
from vbfml.training.util import save
from vbfml.training.data import TrainingLoader


def keras_model_compare(model1, model2):
    for l1, l2 in zip(model1.layers, model2.layers):
        if not l1.get_config() == l2.get_config():
            return False
    return True


class TestTrainingLoader(TestCase):
    def setUp(self):
        self.wdir = make_tmp_dir()

        self.training_sequence_file = os.path.join(self.wdir, "training_sequence.pkl")
        self.validation_sequence_file = os.path.join(
            self.wdir, "validation_sequence.pkl"
        )
        self.feature_file = os.path.join(self.wdir, "features.pkl")
        self.history_file = os.path.join(self.wdir, "history.pkl")
        self.feature_scaler_file = os.path.join(self.wdir, "feature_scaler.pkl")

        self.model = sequential_dense_model(
            n_features=1,
            n_layers=1,
            n_nodes=[1],
            n_classes=1,
        )

        self.model.save(self.wdir)
        save("training_sequence", self.training_sequence_file)
        save("validation_sequence", self.validation_sequence_file)
        save("features", self.feature_file)
        save("history", self.history_file)
        save("feature_scaler", self.feature_scaler_file)

    def test_loader(self):
        loader = TrainingLoader(self.wdir)
        self.assertTrue(keras_model_compare(loader.get_model(), self.model))
        self.assertEqual(loader.get_sequence("training"), "training_sequence")
        self.assertEqual(loader.get_sequence("validation"), "validation_sequence")
        self.assertEqual(loader.get_features(), "features")
        self.assertEqual(loader.get_history(), "history")
        self.assertEqual(loader.get_feature_scaler(), "feature_scaler")
