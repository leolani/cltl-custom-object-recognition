from collections import Counter
from random import choice
import logging

from cltl.combot.event.emissor import LeolaniContext

logger = logging.getLogger(__name__)
from cltl.visual_responder.api import VisualResponder

class VisualResponderImpl(VisualResponder):
    SEE_OBJECT = [
        "what do you see",
        "what can you see",
        "what did you see",
        "what have you seen"
    ]

    SEE_PERSON = [
        "who do you see",
        "who can you see",
    ]

    SEE_PERSON_ALL = [
        "who did you see",
        "who have you seen"
    ]

    SEE_SPECIFIC = [
        "do you see ",
        "can you see ",
        "where is the "
    ]

    I_SEE = [
        "I see",
        "I can see",
        "I think I see",
        "I observe",
    ]

    I_SAW = [
        "I saw",
        "I have seen",
        "I think I observed"
    ]

    NO_OBJECT = [
        "I don't see anything",
        "I don't see any object",
    ]

    NO_PEOPLE = [
        "I don't see anybody I know",
        "I don't see familiar faces",
        "I cannot identify any of my friends",
    ]

    STRANGERS = [
        "persons I do not know",
        "strangers"
    ]

    def __init__(self):
        self.started = False

    # TODO use the confidence scores from the return in the output
    def respond(self, statement: str, context: dict) -> str:
        logger.info("Visual responder checking out:", statement, context)
        if context:
            counts = ', '.join([f"{count} {label}" for label, count in context.items()])
            return f"{choice(self.I_SAW)} {counts}"
        else:
            return choice(self.NO_OBJECT)

#    def _point_to_objects(self, app, obj):
#        app.say("I can see {}".format(self._insert_a_an(obj.name)))
#        app.motion.point(obj.direction, speed=0.2)
#        app.motion.look(obj.direction, speed=0.1)
#        app.say("There it is!!")
