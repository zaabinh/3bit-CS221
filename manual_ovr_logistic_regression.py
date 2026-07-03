import numpy as np
from typing import Any, Dict, Optional
# IMPORT THÊM 2 CLASS NÀY
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse as sp


# Kế thừa thêm BaseEstimator và ClassifierMixin
class ManualLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Cài đặt Logistic Regression từ đầu sử dụng NumPy và thuật toán Gradient Descent.
    Tương thích hoàn toàn với hệ sinh thái scikit-learn (OneVsRestClassifier, GridSearchCV, v.v.)
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[Any] = None,
        max_iter: int = 1000,
        lr: float = 0.01,
        random_state: int = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.w_ = None
        self.b_ = None
        self.classes_ = None 

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if sp.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Lưu thông tin class định danh theo chuẩn sklearn
        self.classes_ = np.unique(y)

        np.random.seed(self.random_state)
        self.w_ = np.random.randn(n_features) * 0.01
        self.b_ = 0.0

        sample_weight = np.ones(n_samples)
        if self.class_weight == "balanced":
            recip_freq = n_samples / (len(self.classes_) * np.bincount(y))
            weight_dict = dict(zip(self.classes_, recip_freq))
            sample_weight = np.array([weight_dict[val] for val in y])
        elif isinstance(self.class_weight, dict):
            sample_weight = np.array([self.class_weight.get(val, 1.0) for val in y])

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.w_) + self.b_
            y_pred = self._sigmoid(linear_model)
            error = (y_pred - y) * sample_weight

            dw = (1 / n_samples) * np.dot(X.T, error) + (1 / self.C) * self.w_
            db = (1 / n_samples) * np.sum(error)

            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if sp.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        linear_model = np.dot(X, self.w_) + self.b_
        prob_class_1 = self._sigmoid(linear_model)
        prob_class_0 = 1 - prob_class_1
        return np.column_stack((prob_class_0, prob_class_1))

    def predict(self, X) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return np.where(prob >= 0.5, 1, 0)
    
