import logging
from collections import Counter, deque
from typing import Deque, Dict

from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import TextSignal, class_type
from cltl.visual_responder.api import VisualResponder, ImageAnnotation, DEFAULT_COUNT_THRESHOLD
from cltl.combot.infra.event.util import extract_scenario_id
from cltl.object_recognition.api import Object, SCENE_TYPE, SCENE_DESCRIPTION_TYPE
logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'

# Default number of image annotations that reflect a change kept per scenario
DEFAULT_HISTORY_SIZE = 5


class VisualResponderService:
    @classmethod
    def from_config(cls, responder: VisualResponder, event_bus: EventBus,
                    resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.visual-responder")

        return cls(config.get("text_input"),  config.get("object_input"),
                   config.get("topic_output"),
                   responder,
                   event_bus, resource_manager,
                   count_threshold=int(config.get("count_threshold")) if "count_threshold" in config
                                   else DEFAULT_COUNT_THRESHOLD,
                   history_size=int(config.get("history_size")) if "history_size" in config
                                else DEFAULT_HISTORY_SIZE)

    def __init__(self, input_text: str, input_object: str, output_topic: str,
                 responder: VisualResponder,
                 event_bus: EventBus, resource_manager: ResourceManager,
                 count_threshold: int = DEFAULT_COUNT_THRESHOLD, history_size: int = DEFAULT_HISTORY_SIZE):
        self._responder = responder
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._input_text = input_text
        self._input_object = input_object
        self._output_topic = output_topic
        self._count_threshold = count_threshold
        self._history_size = history_size
        self._topic_worker = None
        # Per scenario the last history_size image annotations that differ from their predecessor, oldest first
        self._context: Dict[str, Deque[ImageAnnotation]] = {}

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_text, self._input_object], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process, buffer_size=64,
                                         name=self.__class__.__name__)

        # provided_topics = list(filter(None, [self._output_topic, self._forward_topic]))
        # self._topic_worker = TopicWorker([self._input_topic, self._scenario_topic], self._event_bus,
        #                                  resource_manager=self._resource_manager, processor=self._process,
        #                                  name=self.__class__.__name__)

        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass
        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event):
        logger.info("VISUAL RESPONDER HERE: Should I respond to this? %s", event.metadata.topic)
        if event.metadata.topic == self._input_object:
            self._process_object(event)
        elif event.metadata.topic == self._input_text:
            self._process_text(event)
        else:
            raise ValueError("Unexpected topic " + event.metadata.topic)

    def _process_text(self, event: Event[TextSignalEvent]):
        scenario_id = extract_scenario_id(event)
        history = list(self._context.get(scenario_id, ()))
        logger.info("VISUAL RESPONDER HERE: Should I respond to this? %s, %s", event.payload.signal.text, history)
        response = self._responder.respond(event.payload.signal.text, history)
        if response:
            about_event = self._create_payload(response, scenario_id)
            self._event_bus.publish(self._output_topic, Event.for_payload(about_event, source=event))
            logger.info("Visual responder answered %s with %s", event.payload.signal.text, response)

    def _process_object(self, event: Event[TextSignalEvent]):
        scenario_id = extract_scenario_id(event)
        annotation = self._to_image_annotation(event)

        history = self._context.setdefault(scenario_id, deque(maxlen=self._history_size))
        if annotation.is_change_from(history, self._count_threshold):
            # With maxlen, appending to a full deque drops the oldest annotation
            history.append(annotation)
            logger.info("Image changed for scenario %s: %s", scenario_id, annotation)

    @staticmethod
    def _to_image_annotation(event) -> ImageAnnotation:
        objects = [annotation.value
                   for mention in event.payload.mentions
                   for annotation in mention.annotations
                   if annotation.type == class_type(Object) and annotation.value]

        scene = next((obj.label for obj in objects if obj.type == SCENE_TYPE), None)
        scene_description = next((obj.label for obj in objects if obj.type == SCENE_DESCRIPTION_TYPE), None)
        labels = Counter(obj.label for obj in objects if obj.type not in (SCENE_TYPE, SCENE_DESCRIPTION_TYPE))

        return ImageAnnotation(scene, scene_description, labels)

    def _create_payload(self, response, scenario_id):
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)
        # for_agent, not for_speaker: annotates the signal as coming from the
        # agent, which is what puts the reply on the agent's side of the chat.
        return TextSignalEvent.for_agent(signal)

