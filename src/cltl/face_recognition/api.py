import abc
import dataclasses
from typing import List, Optional, Iterable, Tuple

import numpy as np
from cltl.backend.api.camera import Bounds
from emissor.representation.entity import Gender


@dataclasses.dataclass
class Face:
    """
    Information about a Face.

    Includes a vector representation of the face and optional meta information.
    """
    # TODO switch to np.typing.ArrayLike
    embedding: np.ndarray
    gender: Optional[Gender]
    age: Optional[int]


class FaceDetector(abc.ABC):
    """
    Detect faces in an image.
    """

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Face], Iterable[Bounds]]:
        """
        Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray
            The binary image.

        Returns
        -------
        Iterable[Face]
            The faces detected in the image.
        Iterable[Bounds]
            The positions of the detected faces in the image.
        """
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass