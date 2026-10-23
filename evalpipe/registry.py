"""Models the suite may evaluate, each with a strict hyperparameter contract."""
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled hyperparameter is an error, not a silent default


class LogRegParams(Params):
    C: float = Field(1.0, gt=0)
    max_iter: int = Field(300, ge=10, le=5000)


class LinearSVMParams(Params):
    C: float = Field(0.1, gt=0)


class ForestParams(Params):
    n_estimators: int = Field(200, ge=1, le=2000)
    max_depth: int | None = Field(None, ge=1)


class KnnParams(Params):
    n_neighbors: int = Field(5, ge=1, le=100)


class MlpParams(Params):
    hidden: list[int] = Field([256], min_length=1, max_length=4)
    max_iter: int = Field(60, ge=1, le=1000)


@dataclass(frozen=True)
class ModelEntry:
    params: type[Params]
    build: Callable[[Params, int], object]
    has_proba: bool


REGISTRY: dict[str, ModelEntry] = {
    "logreg": ModelEntry(LogRegParams, lambda p, s: LogisticRegression(C=p.C, max_iter=p.max_iter), True),
    "linear_svm": ModelEntry(LinearSVMParams, lambda p, s: LinearSVC(C=p.C, random_state=s), False),
    "random_forest": ModelEntry(ForestParams, lambda p, s: RandomForestClassifier(
        n_estimators=p.n_estimators, max_depth=p.max_depth, random_state=s, n_jobs=1), True),
    "knn": ModelEntry(KnnParams, lambda p, s: KNeighborsClassifier(n_neighbors=p.n_neighbors), True),
    "mlp": ModelEntry(MlpParams, lambda p, s: MLPClassifier(
        hidden_layer_sizes=tuple(p.hidden), max_iter=p.max_iter, early_stopping=True, random_state=s), True),
}
