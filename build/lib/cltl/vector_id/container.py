import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton
from cltl.vector_id.api import VectorIdentity
from cltl.vector_id.clusterid import ClusterIdentity
from cltl.vector_id.service import VectorIdService
from cltl.face_recognition.service import FaceRecognitionService

logger = logging.getLogger(__name__)

class VectorIdContainer(InfraContainer):
    @property
    @singleton
    def vector_id(self) -> VectorIdentity:
        config = self.config_manager.get_config("cltl.vector_id.agg")

        return ClusterIdentity.agglomerative(0, config.get_float("distance_threshold"), config.get("storage_path"))

    @property
    @singleton
    def vector_id_service(self) -> FaceRecognitionService:
        return VectorIdService.from_config(self.vector_id, self.event_bus,
                                           self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Vector ID")
        super().start()
        self.vector_id_service.start()

    def stop(self):
        try:
            logger.info("Stop Vector ID")
            self.vector_id_service.stop()
        finally:
            super().stop()

