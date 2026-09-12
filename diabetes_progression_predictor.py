import argparse
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DataLoader:
    """Loads the dataset and exposes it as a pandas DataFrame."""

    def __init__(self):
        self.__data = None  # encapsulated: accessed only through methods

    def load_data(self):
        """Loads the diabetes progression dataset into a DataFrame."""
        diabetes = load_diabetes(as_frame=True)
        self.__data = diabetes.frame
        return self.__data

    def get_data(self):
        """Returns the loaded dataset."""
        return self.__data


class DataPreprocessor:
    """Handles validation, feature/target splitting, and train/test data."""

    def __init__(self, target_column):
        self.__target_column = target_column
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None

    def split_features_target(self, data):
        """Validates the data, then separates it into features (X) and target (y)."""
        if self.__target_column not in data.columns:
            raise ValueError(f"Target column '{self.__target_column}' not found in data.")
        if data.isnull().values.any():
            raise ValueError("Dataset contains missing values; please clean it before training.")
        X = data.drop(columns=[self.__target_column])
        y = data[self.__target_column]
        return X, y

    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        """Splits data into training and testing sets."""
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = (
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        )
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test


class BaseModel(ABC):
    """Abstract base class defining the common interface for all models."""

    def __init__(self):
        self._model = None
        self._is_trained = False

    @abstractmethod
    def _build_model(self):
        """Creates and returns the underlying scikit-learn estimator."""
        raise NotImplementedError

    def train(self, X_train, y_train):
        """Builds a fresh estimator and trains it on the given data."""
        self._model = self._build_model()
        self._model.fit(X_train, y_train)
        self._is_trained = True

    def predict(self, X_test):
        """Generates predictions for new/unseen data."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        return self._model.predict(X_test)

    def get_coefficients(self):
        """Returns the learned coefficients of the model."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before accessing coefficients.")
        return self._model.coef_

    def save_model(self, filepath):
        """Saves the trained model to disk using joblib."""
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self._model, filepath)

    def load_model(self, filepath):
        """Loads a previously saved model from disk."""
        self._model = joblib.load(filepath)
        self._is_trained = True

    @property
    def name(self):
        """Returns a human-readable name for the model."""
        return self.__class__.__name__


class LinearModel(BaseModel):
    """Ordinary Linear Regression model."""

    def _build_model(self):
        """Creates a plain Linear Regression estimator."""
        return LinearRegression()


class RidgeModel(BaseModel):
    """Ridge Regression model (linear regression with L2 regularization)."""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.__alpha = alpha

    def _build_model(self):
        """Creates a Ridge Regression estimator with the configured alpha."""
        return Ridge(alpha=self.__alpha)


class ModelEvaluator:
    """Evaluates model predictions against true values."""

    def __init__(self, y_true, y_pred):
        self.__y_true = y_true
        self.__y_pred = y_pred

    def mean_squared_error(self):
        """Calculates the mean squared error of the predictions."""
        return mean_squared_error(self.__y_true, self.__y_pred)

    def mean_absolute_error(self):
        """Calculates the mean absolute error of the predictions."""
        return mean_absolute_error(self.__y_true, self.__y_pred)

    def r2_score(self):
        """Calculates the R-squared score of the predictions."""
        return r2_score(self.__y_true, self.__y_pred)

    def summary(self):
        """Prints a short, readable evaluation summary."""
        print("Model Evaluation")
        print(f"  Mean Squared Error:  {self.mean_squared_error():.4f}")
        print(f"  Mean Absolute Error: {self.mean_absolute_error():.4f}")
        print(f"  R-squared Score:     {self.r2_score():.4f}")

    def plot_predictions(self, filepath="predictions_plot.png"):
        """Saves a scatter plot comparing actual vs. predicted values."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib is not installed; skipping plot.")
            return
        y_true = np.array(self.__y_true)
        y_pred = np.array(self.__y_pred)
        lower, upper = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([lower, upper], [lower, upper], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.savefig(filepath)
        plt.close()
        print(f"Prediction plot saved to {filepath}")


class DiseaseProgressionPipeline:
    """Orchestrates loading, preprocessing, training, and evaluation."""

    def __init__(self, model, target_column="target", test_size=0.2):
        self.__loader = DataLoader()
        self.__preprocessor = DataPreprocessor(target_column)
        self.__model = model  # polymorphism: any BaseModel subclass works here
        self.__test_size = test_size

    def run(self, use_cross_validation=False, save_path=None, plot=False):
        """Runs the full pipeline end-to-end and prints results."""
        data = self.__loader.load_data()
        X, y = self.__preprocessor.split_features_target(data)
        X_train, X_test, y_train, y_test = self.__preprocessor.split_train_test(
            X, y, test_size=self.__test_size
        )

        self.__model.train(X_train, y_train)
        predictions = self.__model.predict(X_test)

        print(f"Model used: {self.__model.name}")
        evaluator = ModelEvaluator(y_test, predictions)
        evaluator.summary()

        if use_cross_validation:
            self.__run_cross_validation(X, y)

        print("\nSample Predictions vs Actual:")
        comparison = pd.DataFrame({
            "Actual": np.array(y_test)[:5],
            "Predicted": np.round(predictions[:5], 2),
        })
        print(comparison.to_string(index=False))

        if save_path:
            self.__model.save_model(save_path)
            print(f"\nModel saved to {save_path}")

        if plot:
            evaluator.plot_predictions()

    def __run_cross_validation(self, X, y, cv=5):
        """Runs k-fold cross-validation and prints the average R2 score."""
        fresh_estimator = self.__model._build_model()
        scores = cross_val_score(fresh_estimator, X, y, cv=cv, scoring="r2")
        print(f"\nCross-Validation R2 (cv={cv}): mean={scores.mean():.4f}, std={scores.std():.4f}")


def build_model(model_name, alpha):
    """Creates a model instance based on the requested model name."""
    if model_name == "ridge":
        return RidgeModel(alpha=alpha)
    return LinearModel()


def parse_arguments():
    """Parses command-line arguments for configuring the pipeline."""
    parser = argparse.ArgumentParser(description="Diabetes Progression Predictor")
    parser.add_argument("--model", choices=["linear", "ridge"], default="linear",
                         help="Which model to use (default: linear)")
    parser.add_argument("--alpha", type=float, default=1.0,
                         help="Regularization strength for the Ridge model")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Proportion of data used for testing (default: 0.2)")
    parser.add_argument("--cross-validate", action="store_true",
                         help="Also run k-fold cross-validation")
    parser.add_argument("--save-model", type=str, default=None,
                         help="File path to save the trained model (e.g. model.joblib)")
    parser.add_argument("--plot", action="store_true",
                         help="Save a scatter plot of actual vs predicted values")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    selected_model = build_model(args.model, args.alpha)
    pipeline = DiseaseProgressionPipeline(selected_model, test_size=args.test_size)
    pipeline.run(
        use_cross_validation=args.cross_validate,
        save_path=args.save_model,
        plot=args.plot,
    )
