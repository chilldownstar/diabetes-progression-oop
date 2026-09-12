# Diabetes Progression Predictor (OOP)

A small ML project I built to practice OOP in Python and combine it with ML concepts — trains a regression model to predict diabetes progression using scikit-learn.

## What it demonstrates

- **Classes & Objects** — each step of the ML workflow (loading, preprocessing, modeling, evaluation) is its own class.
- **Encapsulation** — internal state (`__data`, `_is_trained`, etc.) is private and only accessed through methods.
- **Constructors** — every class sets up its own state in `__init__`.
- **Inheritance** — `LinearModel` and `RidgeModel` both inherit from a shared `BaseModel`.
- **Polymorphism** — the pipeline trains/predicts through any `BaseModel` subclass without knowing its concrete type.

## Project structure

```
diabetes_progression_predictor.py       # main pipeline and all classes
test_diabetes_progression_predictor.py  # unit tests
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run with default settings (Linear Regression):

```bash
python diabetes_progression_predictor.py
```

Run with more options (Ridge Regression, cross-validation, save the trained model, and plot predictions):

```bash
python diabetes_progression_predictor.py --model ridge --alpha 0.5 --cross-validate --save-model model.joblib --plot
```

Available options:

| Flag | Description |
|---|---|
| `--model` | `linear` (default) or `ridge` |
| `--alpha` | Regularization strength for Ridge |
| `--test-size` | Fraction of data used for testing (default: 0.2) |
| `--cross-validate` | Also run 5-fold cross-validation |
| `--save-model` | Path to save the trained model (e.g. `model.joblib`) |
| `--plot` | Save a scatter plot comparing actual vs. predicted values |

## Running tests

```bash
python -m unittest test_diabetes_progression_predictor.py -v
```

## Dataset

Uses scikit-learn's built-in [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) — no download required.
