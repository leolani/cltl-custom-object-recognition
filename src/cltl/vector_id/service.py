import logging

from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker

from cltl.vector_id.api import VectorIdentity
from cltl.face_recognition.schema import FaceRecognitionEvent
from cltl.vector_id.schema import VectorIdentityEvent

logger = logging.getLogger(__name__)


class VectorIdService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, vector_id: VectorIdentity, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.vector_id.events")

        return cls(config.get("face_topic"), config.get("id_topic"), vector_id, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str,  vector_id: VectorIdentity,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._vector_id = vector_id

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

    def start(self, timeout=30):
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

    def _process(self, event: Event[FaceRecognitionEvent]):
        representations = [annotation.value.embedding
                           for mention in event.payload.mentions
                           for annotation in mention.annotations
                           if annotation.value]

        if representations:
            segments = [segment
                        for mention in event.payload.mentions
                        for segment in mention.segment]

            ids = self._vector_id.add(representations)

            id_payload = VectorIdentityEvent.create(segments, ids)
            self._event_bus.publish(self._output_topic, Event.for_payload(id_payload))
