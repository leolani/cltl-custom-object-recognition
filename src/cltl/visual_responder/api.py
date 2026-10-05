import abc
from typing import List, Optional

from cltl.situation_awareness.api import ImageAnnotation


class VisualResponder(abc.ABC):
    def respond(self, statement: str, history: List[ImageAnnotation]) -> Optional[str]:
        """
        Parameters
        ----------
        statement : str
            The utterance to respond to.
        history : List[ImageAnnotation]
            The most recent image annotations that reflect a change, oldest first.

        Returns
        -------
        Optional[str]
            The response, or None if the statement should not be responded to.
        """
        raise NotImplementedError("")
