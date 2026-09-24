import abc
import numpy as np


class VectorIdentity(abc.ABC):
    def register(self, id: str, representations: np.ndarray) -> str:
        raise NotImplementedError("")

    def add(self, representations: np.ndarray) -> str:
        raise NotImplementedError("")

    def resolve_id(self, representation: np.ndarray) -> str:
        raise NotImplementedError("")


