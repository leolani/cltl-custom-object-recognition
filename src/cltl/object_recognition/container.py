import logging
from sys import implementation

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.object_recognition.api import ObjectDetector
from cltl.object_recognition.ollama_proxy import OllamaObjectDetectorProxy
from cltl.object_recognition.proxy import ObjectDetectorProxy
from cltl.object_recognition.service import ObjectRecognitionService

logger = logging.getLogger(__name__)


class ObjectRecognitionContainer(InfraContainer):
    @property
    @singleton
    def object_detector(self) -> ObjectDetector:
        config = self.config_manager.get_config("cltl.object_recognition")
        implementation = config.get("implementation")
        if not implementation:
             logger.warning("No ObjectDetector configured")
             return False

        if implementation != "proxy" and implementation != "vlm":
            raise ValueError("Unknown ObjectRecognition implementation: " + implementation)

        logger.info("Starting Object Recognition: %s", implementation)

        if implementation=="proxy":
             ### needed for Yolo docker service
             config = self.config_manager.get_config("cltl.object_recognition.proxy")
             start_infra = config.get_boolean("start_infra")
             detector_url = config.get("detector_url") if "detector_url" in config else None
             logger.info("Starting Object Recognition Proxy")
             return ObjectDetectorProxy(start_infra, detector_url)
        elif implementation=="vlm":
             config = self.config_manager.get_config("cltl.object_recognition.vlm")
             model = config.get("model")
             logger.info("Starting Object Recognition VLM")
             return OllamaObjectDetectorProxy(model=model)

    @property
    @singleton
    def object_recognition_service(self) -> ObjectRecognitionService:
        if self.object_detector:
            return ObjectRecognitionService.from_config(self.object_detector, self.event_bus,
                                                        self.resource_manager, self.config_manager)
        else:
            logger.warning("No ObjectRecognitionService configured")
            return False

    def start(self):
        super().start()
        if self.object_recognition_service:
            logger.info("Start Object Recognition")
            self.object_recognition_service.start()

    def stop(self):
        try:
            if self.object_recognition_service:
                logger.info("Stop Object Recognition")
                self.object_recognition_service.stop()
        finally:
            super().stop()

