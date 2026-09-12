import unittest
import numpy as np
import pandas as pd
from diabetes_progression_predictor import (
    DataPreprocessor,
    LinearModel,
    RidgeModel,
    ModelEvaluator,
)


class TestDataPreprocessor(unittest.TestCase):
    """Tests for feature/target splitting and input validation."""

    def setUp(self):
        self.data = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [4.0, 3.0, 2.0, 1.0],
            "target": [10.0, 20.0, 30.0, 40.0],
        })
        self.preprocessor = DataPreprocessor(target_column="target")

    def test_split_features_target(self):
        """Splitting should separate features from the target column correctly."""
        X, y = self.preprocessor.split_features_target(self.data)
        self.assertNotIn("target", X.columns)
        self.assertEqual(len(y), 4)

    def test_missing_target_column_raises_error(self):
        """A missing target column should raise a clear ValueError."""
        preprocessor = DataPreprocessor(target_column="does_not_exist")
        with self.assertRaises(ValueError):
            preprocessor.split_features_target(self.data)

    def test_missing_values_raise_error(self):
        """Missing values in the dataset should raise a ValueError."""
        dirty_data = self.data.copy()
        dirty_data.loc[0, "feature_1"] = np.nan
        with self.assertRaises(ValueError):
            self.preprocessor.split_features_target(dirty_data)


class TestModels(unittest.TestCase):
    """Tests that both model types train and predict without errors."""

    def setUp(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.rand(20, 3)
        self.y_train = rng.rand(20)
        self.X_test = rng.rand(5, 3)

    def test_linear_model_train_predict(self):
        """LinearModel should train and produce the right number of predictions."""
        model = LinearModel()
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 5)

    def test_ridge_model_train_predict(self):
        """RidgeModel should train and produce the right number of predictions."""
        model = RidgeModel(alpha=0.5)
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 5)

    def test_predict_before_train_raises_error(self):
        """Calling predict before train should raise a RuntimeError."""
        model = LinearModel()
        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)

    def test_model_name_reflects_class(self):
        """The name property should return the concrete subclass name."""
        self.assertEqual(LinearModel().name, "LinearModel")
        self.assertEqual(RidgeModel().name, "RidgeModel")


class TestModelEvaluator(unittest.TestCase):
    """Tests for evaluation metric calculations."""

    def test_perfect_predictions_give_zero_error(self):
        """Identical actual and predicted values should give zero error and R2 of 1."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        evaluator = ModelEvaluator(y_true, y_pred)
        self.assertAlmostEqual(evaluator.mean_squared_error(), 0.0)
        self.assertAlmostEqual(evaluator.mean_absolute_error(), 0.0)
        self.assertAlmostEqual(evaluator.r2_score(), 1.0)


if __name__ == "__main__":
    unittest.main()
