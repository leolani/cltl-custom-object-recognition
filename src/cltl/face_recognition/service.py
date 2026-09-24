import logging
from typing import Callable

from cltl.backend.source.client_source import ClientImageSource
from cltl.backend.spi.image import ImageSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.event.emissor import ImageSignalEvent

from cltl.face_recognition.api import FaceDetector
from cltl.face_recognition.schema import FaceRecognitionEvent

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, face_detector: FaceDetector, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.face_recognition.events")

        def image_loader(url) -> ImageSource:
            return ClientImageSource.from_config(config_manager, url)


        return cls(config.get("image_topic"), config.get("face_topic"), face_detector, image_loader,
                   event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, face_detector: FaceDetector,
                 image_loader: Callable[[str], ImageSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._face_detector = face_detector
        self._image_loader = image_loader

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

    def start(self, timeout=30):
        self._face_detector.__enter__()
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None
        self._face_detector.__exit__(None, None, None)

    def _process(self, event: Event[ImageSignalEvent]):
        image_location = event.payload.signal.files[0]

        with self._image_loader(image_location) as source:
            image = source.capture()
        faces, bounds = self._face_detector.detect(image.image)

        face_event = FaceRecognitionEvent.create(event.payload.signal, faces, bounds)
        self._event_bus.publish(self._output_topic, Event.for_payload(face_event))