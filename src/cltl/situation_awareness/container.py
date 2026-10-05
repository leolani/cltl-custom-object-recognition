import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.situation_awareness.api import DEFAULT_COUNT_THRESHOLD
from cltl.situation_awareness.history import VisualHistory, DEFAULT_HISTORY_SIZE
from cltl.situation_awareness.knowledge_graph import SceneKnowledgeGraph, connect_brain, \
    DEFAULT_SIMILARITY_THRESHOLD
from cltl.situation_awareness.service import SituationAwarenessService

logger = logging.getLogger(__name__)


class SituationAwarenessContainer(InfraContainer):
    @property
    @singleton
    def visual_history(self) -> VisualHistory:
        config = self.config_manager.get_config("cltl.situation-awareness")
        count_threshold = int(config.get("count_threshold")) if "count_threshold" in config else DEFAULT_COUNT_THRESHOLD
        history_size = int(config.get("history_size")) if "history_size" in config else DEFAULT_HISTORY_SIZE

        return VisualHistory(count_threshold, history_size)

    @property
    @singleton
    def scene_knowledge_graph(self) -> SceneKnowledgeGraph:
        config = self.config_manager.get_config("cltl.situation-awareness")
        kg_address = config.get("kg_address") if "kg_address" in config else None
        if not kg_address or kg_address.startswith("$"):
            logger.info("No knowledge graph configured, scenes are not pushed")
            # Disabled rather than None, as a singleton cannot be None
            return SceneKnowledgeGraph(None)

        log_dir = config.get("kg_log_dir") if "kg_log_dir" in config else "kg_logs"
        similarity_threshold = float(config.get("similarity_threshold")) if "similarity_threshold" in config \
            else DEFAULT_SIMILARITY_THRESHOLD
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {similarity_threshold}")
        source = config.get("source") if "source" in config else "front-camera"
        # The context of the interaction is pushed only if a place is configured
        context = {key: config.get(key) for key in ("place", "country", "region", "city") if key in config} \
            if "place" in config else None

        logger.info("Push scenes to %s with similarity_threshold %s", kg_address, similarity_threshold)

        return SceneKnowledgeGraph(lambda: connect_brain(kg_address, log_dir), similarity_threshold, source, context)

    @property
    @singleton
    def situation_awareness_service(self) -> SituationAwarenessService:
        return SituationAwarenessService.from_config(self.visual_history, self.scene_knowledge_graph,
                                                     self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start SituationAwareness")
        super().start()
        self.situation_awareness_service.start()

    def stop(self):
        try:
            logger.info("Stop SituationAwareness")
            self.situation_awareness_service.stop()
        finally:
            super().stop()
