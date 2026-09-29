import abc
import dataclasses
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cltl.visual_responder.labels import by_category, labels_by_category

# Differences in the count of an object label up to this value are ignored when comparing annotations
DEFAULT_COUNT_THRESHOLD = 1


@dataclasses.dataclass
class ImageAnnotation:
    """
    Summary of the object recognition output for a single image.

    Annotations are compared on the counts per object category (see labels.CATEGORIES),
    such that e.g. "man" and "woman" both count as "person", and counts that differ by
    at most a threshold are considered equal, as a VLM varies in the words and counts it
    reports for images of the same view. The scene_description is not compared as it is
    phrased differently for every image.
    """
    scene: Optional[str]
    scene_description: Optional[str]
    objects: Counter = dataclasses.field(default_factory=Counter)
    # Set once the changes in this annotation were reported, so they are not reported again
    reported: bool = dataclasses.field(default=False, compare=False)

    def is_change_from(self, history: Sequence["ImageAnnotation"],
                       count_threshold: int = DEFAULT_COUNT_THRESHOLD) -> bool:
        if not history or self.scene != history[-1].scene:
            return True

        # Compared to the most recent annotation only, so objects that are seen again are part of the
        # history, even though object_changes does not report them as new.
        appeared, disappeared = self.object_changes([history[-1]], count_threshold)
        return bool(appeared or disappeared)

    def object_changes(self, history: Sequence["ImageAnnotation"],
                       count_threshold: int = DEFAULT_COUNT_THRESHOLD) -> Tuple[Counter, Counter]:
        """
        Returns the objects that appeared and disappeared compared to the history of
        previous annotations, oldest first, with the difference in count per object.

        Objects appeared if their count is higher than in the most recent annotation and
        does not match the count in any annotation in the history, so objects that are
        seen again after they were missed are not reported as new. Objects disappeared if
        their count is lower than in the most recent annotation and does not match it.
        Counts match if the object is present in both and the counts differ by at most
        count_threshold.

        Objects are compared per category and reported with their label if it is the only
        label of the category, otherwise with the category name.
        """
        current = by_category(self.objects)
        previous = [by_category(annotation.objects) for annotation in history]
        recent = previous[-1] if previous else Counter()

        def matches(count, other_count):
            return count > 0 and other_count > 0 and abs(count - other_count) <= count_threshold

        appeared, disappeared = Counter(), Counter()
        for cat in set(current) | set(recent):
            count, recent_count = current[cat], recent[cat]
            if count > recent_count and not any(matches(count, counts[cat]) for counts in previous):
                appeared[cat] = count - recent_count
            elif count < recent_count and not matches(count, recent_count):
                disappeared[cat] = recent_count - count

        return (self._to_labels(appeared, labels_by_category(self.objects)),
                self._to_labels(disappeared, labels_by_category(history[-1].objects) if history else {}))

    @staticmethod
    def _to_labels(counts: Counter, labels: Dict[str, Set[str]]) -> Counter:
        return Counter({next(iter(labels[cat])) if len(labels.get(cat, ())) == 1 else cat: count
                        for cat, count in counts.items()})


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
