import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton
from cltl.visual_responder.api import VisualResponder
from cltl.visual_responder.visualresponder import VisualResponderImpl
from cltl.visual_responder.service import VisualResponderService
logger = logging.getLogger(__name__)

class VisualResponderContainer(InfraContainer):
    @property
    @singleton
    def visual_responder(self) -> VisualResponder:
        return VisualResponderImpl()

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


