import logging
from typing import List

from cltl.combot.event.emissor import TextSignalEvent, ScenarioStarted, ScenarioStopped, ScenarioEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import TextSignal
from cltl.visual_responder.api import VisualResponder

logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'


class VisualResponderService:
    @classmethod
    def from_config(cls, responder: VisualResponder, event_bus: EventBus,
                    resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.visual-responder")

        return cls(config.get("topic_scenario"), config.get("topic_input"),
                   config.get("topic_response"), config.get("topic_forward"),
                   responder, config.get("intentions", multi=True), config.get("topic_intentions"),
                   event_bus, resource_manager)

    def __init__(self, scenario_topic: str, input_topic: str, response_topic: str, forward_topic: str,
                 responder: VisualResponder,
                 intentions: List[str], intention_topic: str,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._responder = responder

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._scenario_topic = scenario_topic
        self._input_topic = input_topic
        self._response_topic = response_topic
        self._forward_topic = forward_topic

        self._intentions = intentions if intentions else ()
        self._intention_topic = intention_topic if intention_topic else None

        self._topic_worker = None

        self._context = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        provided_topics = list(filter(None, [self._response_topic, self._forward_topic]))
        self._topic_worker = TopicWorker([self._input_topic, self._scenario_topic], self._event_bus, provides=provided_topics,
                                         intentions=self._intentions, intention_topic=self._intention_topic,
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event):
        if event.metadata.topic == self._scenario_topic:
            self._process_scenario(event)
        elif event.metadata.topic == self._input_topic:
            self._process_text(event)
        else:
            raise ValueError("Unexpected topic " + event.metadata.topic)

    def _process_scenario(self, event):
        if event.payload.type in [ScenarioStarted.__name__, ScenarioEvent.__name__]:
            self._context = event.payload.scenario.context
            logger.debug("Updated scenario context to %s", self._context)
        elif event.payload.type == ScenarioStopped.__name__:
            self._context = None
            logger.debug("Stopped scenario %s", event.payload.scenario.id)
        else:
            raise ValueError("Unexpected event type " + event.payload.type)

    def _process_text(self, event: Event[TextSignalEvent]):
        response = self._responder.respond(event.payload.signal.text, self._context)
        if response:
            about_event = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(about_event))
            logger.debug("Answered %s with %s", event.payload.signal.text, response)
        elif self._forward_topic:
            self._event_bus.publish(self._forward_topic, event)
            logger.debug("Forwarded %s to topic %s", event.payload.signal.text, self._forward_topic)

    def _create_payload(self, response):
        scenario_id = self._emissor_client.get_current_scenario_id()
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.for_agent(signal)
