from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def make_three_class_target(y: pd.Series) -> pd.Series:
    """Map original wine quality (3–8) to 3 classes:
    low: 3–4 -> 0, medium: 5–6 -> 1, high: 7–8 -> 2
    """
    def _map(v: int) -> int:
        if v <= 4:
            return 0
        if v <= 6:
            return 1
        return 2
    return y.astype(int).map(_map)

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)

def scale_for_tabnet(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standard scaling (useful for neural/tabnet)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler
