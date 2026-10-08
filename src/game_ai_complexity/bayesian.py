from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


class BayesianLogisticLaplace:
    """Approximate Bayesian logistic regression via Laplace approximation."""

    def __init__(self, prior_precision: float = 1.0, random_state: int = 42):
        self.prior_precision = prior_precision
        self.random_state = random_state
        self.beta_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "BayesianLogisticLaplace":
        model = LogisticRegression(
            C=1.0 / self.prior_precision,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        beta = np.concatenate([model.intercept_, model.coef_.ravel()])
        x_aug = np.column_stack([np.ones(len(x_train)), x_train])
        probs = sigmoid(x_aug @ beta)
        weights = probs * (1.0 - probs)

        hessian = (x_aug.T * weights) @ x_aug
        prior = np.eye(x_aug.shape[1]) * self.prior_precision
        prior[0, 0] = 1e-6
        self.beta_ = beta
        self.covariance_ = np.linalg.pinv(hessian + prior + np.eye(x_aug.shape[1]) * 1e-6)
        return self

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("Model is not fitted")
        x_aug = np.column_stack([np.ones(len(x_values)), x_values])
        positive = sigmoid(x_aug @ self.beta_)
        return np.column_stack([1.0 - positive, positive])

    def probability_interval(self, x_values: np.ndarray, samples: int = 250) -> tuple[np.ndarray, np.ndarray]:
        if self.beta_ is None or self.covariance_ is None:
            raise RuntimeError("Model is not fitted")
        rng = np.random.default_rng(self.random_state)
        draws = rng.multivariate_normal(self.beta_, self.covariance_, size=samples, check_valid="ignore")
        x_aug = np.column_stack([np.ones(len(x_values)), x_values])
        probs = sigmoid(draws @ x_aug.T)
        return np.quantile(probs, 0.05, axis=0), np.quantile(probs, 0.95, axis=0)

