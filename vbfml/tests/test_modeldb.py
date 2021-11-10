from unittest import TestCase
from vbfml.util import vbfml_path, ModelDB


class TestModelDB(TestCase):
    def setUp(self) -> None:
        infile = vbfml_path("config/model_params.yml")
        self.db = ModelDB(infile)
        self.features_to_check = {
            "features": list,
            "dropout": float,
            "train_size": float,
            "batch_size_train": int,
            "batch_size_val": int,
            "batch_buffer_size_train": int,
            "batch_buffer_size_val": int,
        }

    def test_exceptions(self):
        with self.assertRaises(AssertionError):
            wrong_model_name = "NonExistentModel"
            self.db.set_model(wrong_model_name)

        with self.assertRaises(AssertionError):
            self.db.set_model("conv")
            self.db.get("NonExistentFeature")

        self.db.clear_model()

    def test_feature_names(self):
        # Minimally, we'd like to have these features stored in modelDB
        for model in ["dense", "conv"]:
            self.db.set_model(model)
            for key in self.features_to_check.keys():
                self.assertIn(key, self.db.data[model])

        self.db.clear_model()

    def test_return_types(self):
        self.db.set_model("conv")
        for key, type in self.features_to_check.items():
            self.assertIsInstance(self.db.get(key), type)

        self.db.clear_model()
