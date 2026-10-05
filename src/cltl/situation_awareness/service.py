import logging
from typing import Optional

from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.event.util import extract_scenario_id
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker

from cltl.situation_awareness.history import VisualHistory
from cltl.situation_awareness.knowledge_graph import SceneKnowledgeGraph

logger = logging.getLogger(__name__)


class SituationAwarenessService:
    """
    Keeps the VisualHistory up to date with the object recognition events, and pushes the
    scenes to the knowledge graph if one is configured.
    """
    @classmethod
    def from_config(cls, visual_history: VisualHistory, knowledge_graph: Optional[SceneKnowledgeGraph],
                    event_bus: EventBus, resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.situation-awareness")

        return cls(config.get("object_input"), visual_history, knowledge_graph, event_bus, resource_manager)

    def __init__(self, input_object: str, visual_history: VisualHistory,
                 knowledge_graph: Optional[SceneKnowledgeGraph],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._input_object = input_object
        self._visual_history = visual_history
        self._knowledge_graph = knowledge_graph
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._topic_worker = None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_object], self._event_bus,
                                         resource_manager=self._resource_manager, processor=self._process,
                                         buffer_size=64, name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            return
        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event):
        scenario_id = extract_scenario_id(event)
        annotation = VisualHistory.to_image_annotation(event)
        self._visual_history.add(scenario_id, annotation)
        # Every image is checked, so only similarity_threshold regulates what is pushed, independent
        # of the count_threshold of the history
        self._knowledge_graph.push(scenario_id, annotation)
