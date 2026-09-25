from collections import Counter
from random import choice
import abc
from cltl.combot.event.emissor import LeolaniContext
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
    def respond(self, statement: str, scenario_context: LeolaniContext) -> str:
        if not scenario_context:
            return

        object_counts = Counter(scenario_context.objects)
        friends = [agent.name for agent in scenario_context.persons if agent.name]
        unrecognized = len(set([agent.uri for agent in scenario_context.persons if not agent.name]))
        strangers = max(unrecognized, object_counts['person'] - len(friends)) if 'person' in object_counts else unrecognized

        # Enumerate Currently Visible Objects
        if any(question in statement.lower() for question in self.SEE_OBJECT):
            if object_counts:
                counts = ', '.join([f"{count} {label}" for label, count in object_counts.items()])
                return f"{choice(self.I_SAW)} {counts}"
            else:
                return choice(self.NO_OBJECT)

        # Enumerate Currently Visible People
        elif any(question in statement.lower() for question in self.SEE_PERSON_ALL):
            if friends or strangers:
                people = ", ".join(friends)
                people += " and " if (people and strangers) else ""
                people += (str(strangers) + " " + choice(self.STRANGERS)) if strangers else ""

                return f"{choice(self.I_SAW)} {people}"
            else:
                return choice(self.NO_PEOPLE)
        elif any(cue in statement.lower() for cue in self.SEE_SPECIFIC):
            for obj in scenario_context.objects:
                if obj in statement.lower():
                    return f"Yes, {choice(self.I_SAW)} {obj}"

            return f"I cannot see a {statement.strip().split(' ')[-1]}"
        else:
            return None

        # Respond to Individual Object Queries
#        else:
#            for cue in self.SEE_SPECIFIC:
#                if cue in utterance.transcript.lower():
#                    for obj in utterance.context.objects:
#                        if obj.name.lower() in utterance.transcript.lower():
#                            return 1.0, lambda: self._point_to_objects(app, obj)
#
#                    return 1.0, lambda: app.say("I cannot see {}".format(self._insert_a_an(utterance.tokens[-1])))

#    def _point_to_objects(self, app, obj):
#        app.say("I can see {}".format(self._insert_a_an(obj.name)))
#        app.motion.point(obj.direction, speed=0.2)
#        app.motion.look(obj.direction, speed=0.1)
#        app.say("There it is!!")
