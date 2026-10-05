import dataclasses
from collections import Counter
from typing import Dict, Optional, Sequence, Set, Tuple

from cltl.situation_awareness.labels import by_category, labels_by_category

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
    # Provenance of the annotation: the id of the image signal, the image bounds (x0, y0, x1, y1)
    # and the confidence of the scene classification
    image_id: Optional[str] = dataclasses.field(default=None, compare=False)
    region: Optional[Tuple[int, int, int, int]] = dataclasses.field(default=None, compare=False)
    confidence: Optional[float] = dataclasses.field(default=None, compare=False)

    def similarity(self, other: "ImageAnnotation") -> float:
        """
        Similarity of the objects in both annotations between 0 and 1, as the weighted Jaccard
        similarity of the counts per object category, i.e. the sum of the minimum count per
        category divided by the sum of the maximum count. Annotations without objects are
        identical. The scene is not compared.
        """
        current, previous = by_category(self.objects), by_category(other.objects)
        categories = set(current) | set(previous)
        if not categories:
            return 1.0

        return (sum(min(current[cat], previous[cat]) for cat in categories)
                / sum(max(current[cat], previous[cat]) for cat in categories))

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
