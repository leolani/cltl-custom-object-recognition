import logging

from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import TextSignal
from cltl.situation_awareness.history import VisualHistory
from cltl.visual_responder.api import VisualResponder
from cltl.combot.infra.event.util import extract_scenario_id
logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'


class VisualResponderService:
    @classmethod
    def from_config(cls, responder: VisualResponder, visual_history: VisualHistory, event_bus: EventBus,
                    resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.visual-responder")

        return cls(config.get("text_input"), config.get("topic_output"),
                   responder, visual_history,
                   event_bus, resource_manager)

    def __init__(self, input_text: str, output_topic: str,
                 responder: VisualResponder, visual_history: VisualHistory,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._responder = responder
        # Per scenario the image annotations that differ from their predecessor, oldest first,
        # kept up to date by the SituationAwarenessService
        self._visual_history = visual_history
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._input_text = input_text
        self._output_topic = output_topic
        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_text], self._event_bus, provides=[self._output_topic],
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
        if event.metadata.topic == self._input_text:
            self._process_text(event)
        else:
            raise ValueError("Unexpected topic " + event.metadata.topic)

    def _process_text(self, event: Event[TextSignalEvent]):
        scenario_id = extract_scenario_id(event)
        history = self._visual_history.get(scenario_id)
        logger.info("VISUAL RESPONDER HERE: Should I respond to this? %s, %s", event.payload.signal.text, history)
        response = self._responder.respond(event.payload.signal.text, history)
        if response:
            about_event = self._create_payload(response, scenario_id)
            self._event_bus.publish(self._output_topic, Event.for_payload(about_event, source=event))
            logger.info("Visual responder answered %s with %s", event.payload.signal.text, response)

    def _create_payload(self, response, scenario_id):
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)
        # for_agent, not for_speaker: annotates the signal as coming from the
        # agent, which is what puts the reply on the agent's side of the chat.
        return TextSignalEvent.for_agent(signal)

