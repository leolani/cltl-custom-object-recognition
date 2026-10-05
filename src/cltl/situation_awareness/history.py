import logging
import threading
from collections import Counter, deque
from typing import Deque, Dict, List

from emissor.representation.scenario import class_type

from cltl.object_recognition.api import Object, SCENE_TYPE, SCENE_DESCRIPTION_TYPE
from cltl.situation_awareness.api import ImageAnnotation, DEFAULT_COUNT_THRESHOLD

logger = logging.getLogger(__name__)

# Default number of image annotations that reflect a change kept per scenario
DEFAULT_HISTORY_SIZE = 5


class VisualHistory:
    """
    Keeps track per scenario of the scene and the objects that were seen, as the last
    history_size image annotations that differ from their predecessor, oldest first.
    """
    def __init__(self, count_threshold: int = DEFAULT_COUNT_THRESHOLD, history_size: int = DEFAULT_HISTORY_SIZE):
        self._count_threshold = count_threshold
        self._history_size = history_size
        self._context: Dict[str, Deque[ImageAnnotation]] = {}
        # The history is updated and read by different services, each on its own thread
        self._lock = threading.Lock()

    @property
    def count_threshold(self) -> int:
        return self._count_threshold

    def get(self, scenario_id: str) -> List[ImageAnnotation]:
        """The image annotations that reflect a change for the scenario, oldest first."""
        with self._lock:
            return list(self._context.get(scenario_id, ()))

    def add(self, scenario_id: str, annotation: ImageAnnotation) -> bool:
        """Adds the annotation if it reflects a change, returns whether it was added."""
        with self._lock:
            history = self._context.setdefault(scenario_id, deque(maxlen=self._history_size))
            if not annotation.is_change_from(history, self._count_threshold):
                return False

            # With maxlen, appending to a full deque drops the oldest annotation
            history.append(annotation)
        logger.info("Image changed for scenario %s: %s", scenario_id, annotation)

        return True

    def add_event(self, scenario_id: str, event) -> bool:
        """Adds the annotation for an object recognition event, see add."""
        return self.add(scenario_id, self.to_image_annotation(event))

    @staticmethod
    def to_image_annotation(event) -> ImageAnnotation:
        mentions = [(mention, annotation.value)
                    for mention in event.payload.mentions
                    for annotation in mention.annotations
                    if annotation.type == class_type(Object) and annotation.value]
        objects = [obj for _, obj in mentions]

        scene = next((obj.label for obj in objects if obj.type == SCENE_TYPE), None)
        scene_description = next((obj.label for obj in objects if obj.type == SCENE_DESCRIPTION_TYPE), None)
        labels = Counter(obj.label for obj in objects if obj.type not in (SCENE_TYPE, SCENE_DESCRIPTION_TYPE))

        # The scene annotates the whole image, without a scene any mention identifies the image
        scene_mention, scene_obj = next(((mention, obj) for mention, obj in mentions if obj.type == SCENE_TYPE),
                                        (event.payload.mentions[0], None) if event.payload.mentions else (None, None))
        segment = scene_mention.segment[0] if scene_mention and scene_mention.segment else None
        image_id = segment.container_id if segment else None
        region = tuple(segment.bounds) if segment and scene_obj else None
        confidence = scene_obj.confidence if scene_obj else None

        return ImageAnnotation(scene, scene_description, labels,
                               image_id=image_id, region=region, confidence=confidence)
