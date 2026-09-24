import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.face_recognition.api import FaceDetector
from cltl.face_recognition.proxy import FaceDetectorProxy
from cltl.face_recognition.service import FaceRecognitionService

logger = logging.getLogger(__name__)

class FaceRecognitionContainer(InfraContainer):
    @property
    @singleton
    def face_detector(self) -> FaceDetector:
        config = self.config_manager.get_config("cltl.face_recognition")

        implementation = config.get("implementation")
        if not implementation:
            logger.warning("No FaceDetector configured")
            return False
        if implementation != "proxy":
            raise ValueError("Unknown FaceEmotionExtractor implementation: " + implementation)

        config = self.config_manager.get_config("cltl.face_recognition.proxy")
        start_infra = config.get_boolean("start_infra")
        detector_url = config.get("detector_url") if "detector_url" in config else None
        age_gender_url = config.get("age_gender_url") if "age_gender_url" in config else None

        return FaceDetectorProxy(start_infra, detector_url, age_gender_url)

    @property
    @singleton
    def face_recognition_service(self) -> FaceRecognitionService:
        if self.face_detector:
            return FaceRecognitionService.from_config(self.face_detector, self.event_bus,
                                                      self.resource_manager, self.config_manager)
        else:
            return False

    def start(self):
        super().start()
        if self.face_recognition_service:
            logger.info("Start Face Recognition")
            self.face_recognition_service.start()

    def stop(self):
        try:
            if self.face_recognition_service:
                logger.info("Stop Face Recognition")
                self.face_recognition_service.stop()
        finally:
            super().stop()

