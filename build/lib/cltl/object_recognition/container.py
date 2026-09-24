import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.object_recognition.api import ObjectDetector
from cltl.object_recognition.ollama_proxy import OllamaObjectDetectorProxy
from cltl.object_recognition.service import ObjectRecognitionService
from cltl.face_recognition.service import FaceRecognitionService

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
        if implementation != "proxy":
            raise ValueError("Unknown FaceEmotionExtractor implementation: " + implementation)


        #### needed for Yolo docker service
        #config = self.config_manager.get_config("cltl.object_recognition.proxy")
        #start_infra = config.get_boolean("start_infra")
        #detector_url = config.get("detector_url") if "detector_url" in config else None
      #  return ObjectDetectorProxy(start_infra, detector_url)
        return OllamaObjectDetectorProxy()

    @property
    @singleton
    def object_recognition_service(self) -> FaceRecognitionService:
        if self.object_detector:
            return ObjectRecognitionService.from_config(self.object_detector, self.event_bus,
                                                        self.resource_manager, self.config_manager)
        else:
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

