import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton
from cltl.visual_responder.api import VisualResponder, DEFAULT_COUNT_THRESHOLD
from cltl.visual_responder.visualresponder import VisualResponderImpl
from cltl.visual_responder.service import VisualResponderService
logger = logging.getLogger(__name__)

class VisualResponderContainer(InfraContainer):
    @property
    @singleton
    def visual_responder(self) -> VisualResponder:
        config = self.config_manager.get_config("cltl.visual-responder")
        count_threshold = int(config.get("count_threshold")) if "count_threshold" in config else DEFAULT_COUNT_THRESHOLD
        # Comma separated lists, the defaults of VisualResponderImpl are used if not configured
        see_cues = config.get("see_cues", multi=True) if "see_cues" in config else None
        change_cues = config.get("change_cues", multi=True) if "change_cues" in config else None

        return VisualResponderImpl(count_threshold, see_cues, change_cues)

    @property
    @singleton
    def visual_responder_service(self) -> VisualResponderService:
        return VisualResponderService.from_config(self.visual_responder,
                                        self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start VisualResponder")
        super().start()
        self.visual_responder_service.start()

    def stop(self):
        try:
            logger.info("Stop VisualResponder")
            self.visual_responder_service.stop()
        finally:
            super().stop()


