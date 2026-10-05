import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

from cltl.situation_awareness.api import ImageAnnotation
from cltl.situation_awareness.situation_capsule import get_capsule_from_scene, get_context_capsule, get_scene_id

logger = logging.getLogger(__name__)

# Scenes are pushed again only if the similarity of their objects to the last pushed
# annotation of the same scene is below this value, see ImageAnnotation.similarity
DEFAULT_SIMILARITY_THRESHOLD = 0.6

UNKNOWN_SCENE = "unknown scene"


def connect_brain(kg_address: str, log_dir: str):
    # Imported here, so the module runs without cltl.brain when no knowledge graph is configured
    from cltl.brain.long_term_memory import LongTermMemory

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return LongTermMemory(address=kg_address, log_dir=Path(log_dir), clear_all=False)


class SceneKnowledgeGraph:
    """
    Pushes capsules with the scenes seen during an interaction to the knowledge graph.

    Each scene type is a unique scene in the interaction (scenario), with the objects seen in it
    as actors. A scene is only pushed if it was not pushed before in the interaction, or if its
    objects changed, i.e. their similarity to the objects last pushed for the scene is below
    similarity_threshold, to prevent overpopulation of the knowledge graph.
    """
    def __init__(self, brain_factory: Optional[Callable[[], object]], similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 source: str = "front-camera", context: Optional[dict] = None):
        """
        Parameters
        ----------
        brain_factory : Callable[[], LongTermMemory]
            Connects to the knowledge graph, called on first use, see connect_brain. Scenes are
            not pushed if None.
        context : dict
            Keyword arguments for situation_capsule.get_context_capsule, the context of each
            interaction is pushed once if set.
        """
        self._brain_factory = brain_factory
        self._brain = None
        self._similarity_threshold = similarity_threshold
        self._source = source
        self._context = context
        self._lock = threading.Lock()
        # Per scenario and scene the last pushed annotation
        self._pushed: Dict[Tuple[str, str], ImageAnnotation] = {}
        self._contexts: Set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._brain_factory is not None

    def is_change(self, scenario_id: str, annotation: ImageAnnotation) -> bool:
        if not annotation.objects:
            # The knowledge graph does not store a scene without objects
            return False

        last = self._pushed.get((scenario_id, self._scene(annotation)))

        return last is None or annotation.similarity(last) < self._similarity_threshold

    def push(self, scenario_id: str, annotation: ImageAnnotation) -> Optional[dict]:
        """Pushes a capsule for the annotation if the scene changed, returns the capsule or None."""
        if not self.enabled:
            return None

        with self._lock:
            if not self.is_change(scenario_id, annotation):
                return None

            scene = self._scene(annotation)
            capsule = get_capsule_from_scene(scene, get_scene_id(scene, scenario_id), annotation.objects.keys(),
                                             annotation.image_id, scenario_id, self._source,
                                             annotation.confidence if annotation.confidence is not None else 1.0,
                                             annotation.region or (0, 0, 0, 0))
            try:
                brain = self._connect()
                if self._context is not None and scenario_id not in self._contexts:
                    brain.capsule_context(get_context_capsule(scenario_id, **self._context))
                    self._contexts.add(scenario_id)
                brain.capsule_event(capsule, reason_types=True, return_thoughts=False, create_label=True)
            except Exception:
                # Not marked as pushed, so the scene is pushed with the next change
                logger.exception("Failed to push scene %s of scenario %s to the knowledge graph", scene, scenario_id)
                return None

            self._pushed[(scenario_id, scene)] = annotation
            logger.info("Pushed scene %s of scenario %s to the knowledge graph with %s",
                        scene, scenario_id, dict(annotation.objects))

            return capsule

    def _connect(self):
        # A failed connection is not cached, the next push tries again
        if self._brain is None:
            self._brain = self._brain_factory()

        return self._brain

    @staticmethod
    def _scene(annotation: ImageAnnotation) -> str:
        return annotation.scene or UNKNOWN_SCENE
