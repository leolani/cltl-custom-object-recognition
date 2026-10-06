from collections import Counter
from random import choice
import logging
import re
from typing import List, Optional

from cltl.situation_awareness.api import ImageAnnotation, DEFAULT_COUNT_THRESHOLD
from cltl.visual_responder.api import VisualResponder

logger = logging.getLogger(__name__)

class VisualResponderImpl(VisualResponder):
    # Default cues to report what is seen, matched as whole words against the statement with
    # apostrophes removed, see _has_cue. Statements without a cue are not responded to.
    # Can be configured with see_cues in [cltl.visual-responder].
    SEE_OBJECT = [
        "look",
        "tell me what did you see",
        "what do you see",
        "what can you see",
        "what did you see",
        "what have you seen",
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

    # Default cues to report what changed, see SEE_OBJECT. Can be configured with change_cues.
    SEE_CHANGE = [
        "what changed",
        "what has changed",
        "whats changed",
        "what is different",
        "whats different",
        "what is new",
        "whats new",
        "anything new",
        "anything changed",
        "any change",
        "any changes",
        "difference",
        "differences",
    ]

    NO_CHANGE = [
        "Nothing has changed since I started looking",
        "Everything still looks the same to me",
    ]

    # When the changes in the most recent image were already reported
    NO_UPDATE = [
        "Nothing has changed since I last told you",
        "I haven't noticed any change since then",
    ]

    # When the only changes are objects that were seen before
    NOTHING_NEW = [
        "I don't see anything new",
        "Nothing new, just things I have seen before",
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

    def __init__(self, count_threshold: int = DEFAULT_COUNT_THRESHOLD,
                 see_cues: Optional[List[str]] = None, change_cues: Optional[List[str]] = None):
        self._count_threshold = count_threshold
        # Normalized like the statement, so cues can be written with apostrophes
        self._see_cues = [self._normalize(cue) for cue in (see_cues or self.SEE_OBJECT) if cue.strip()]
        self._change_cues = [self._normalize(cue) for cue in (change_cues or self.SEE_CHANGE) if cue.strip()]
        self.started = False

    # TODO use the confidence scores from the return in the output
    def respond(self, statement: str, history: List[ImageAnnotation]) -> Optional[str]:
        """Returns None if the statement contains no cue to report what is seen or what changed."""
        if self._has_cue(statement, self._change_cues):
            return self._describe_change(history) if history else choice(self.NO_OBJECT)

        if self._has_cue(statement, self._see_cues):
            return self._describe(history) if history else choice(self.NO_OBJECT)

        return None

    def _describe(self, history: List[ImageAnnotation]) -> str:
        """
        Describes the most recent annotation, with the object counts if it changed from the
        previous one, otherwise with the scene description.
        """
        annotation = history[-1]
        sentences = []
        if annotation.scene:
            sentences.append(f"This looks like {self._insert_a_an(annotation.scene)}.")

        if annotation.scene_description and not self._has_changed(history):
            sentences.append(annotation.scene_description)
        elif annotation.objects:
            sentences.append(f"{choice(self.I_SEE)} {self._counts(annotation.objects)}.")
        else:
            sentences.append(f"{choice(self.NO_OBJECT)}.")

        return " ".join(sentences)

    def _has_changed(self, history: List[ImageAnnotation]) -> bool:
        if len(history) < 2:
            return False

        *older, previous, last = history
        appeared, disappeared = last.object_changes([*older, previous], self._count_threshold)

        return last.scene != previous.scene or bool(appeared or disappeared)

    def _describe_change(self, history: List[ImageAnnotation]) -> str:
        if history[-1].reported:
            return choice(self.NO_UPDATE)

        history[-1].reported = True
        if len(history) < 2:
            return choice(self.NO_CHANGE)

        *older, previous, last = history
        sentences = []
        if last.scene != previous.scene:
            sentences.append(f"The scene changed from {previous.scene or 'unknown'} to {last.scene or 'unknown'}.")

        appeared, disappeared = last.object_changes([*older, previous], self._count_threshold)
        if appeared:
            sentences.append(f"Now I also see {self._counts(appeared)}.")
        if disappeared:
            sentences.append(f"I no longer see {self._counts(disappeared)}.")

        return " ".join(sentences) if sentences else choice(self.NOTHING_NEW)

    @staticmethod
    def _counts(objects: Counter) -> str:
        return ', '.join(f"{count} {VisualResponderImpl._plural(label)}" if count > 1
                         else VisualResponderImpl._insert_a_an(label)
                         for label, count in objects.most_common())

    IRREGULAR_PLURALS = {
        "man": "men",
        "woman": "women",
        "person": "people",
        "child": "children",
        "foot": "feet",
        "tooth": "teeth",
        "mouse": "mice",
        "goose": "geese",
        "knife": "knives",
        "leaf": "leaves",
        "shelf": "shelves",
        "sheep": "sheep",
        "fish": "fish",
        "deer": "deer",
    }

    @staticmethod
    def _plural(label: str) -> str:
        # Only the last word is inflected, e.g. "coffee cup" -> "coffee cups"
        head, _, word = label.rpartition(" ")
        lower = word.lower()

        if lower in VisualResponderImpl.IRREGULAR_PLURALS:
            plural = VisualResponderImpl.IRREGULAR_PLURALS[lower]
        elif lower in VisualResponderImpl.IRREGULAR_PLURALS.values() \
                or (lower.endswith("s") and not lower.endswith(("ss", "us", "is"))):
            # Already plural, e.g. "glasses", "slippers", "people"
            plural = word
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            plural = word + "es"
        elif lower.endswith("y") and lower[-2:-1] not in ("a", "e", "i", "o", "u"):
            plural = word[:-1] + "ies"
        else:
            plural = word + "s"

        return f"{head} {plural}" if head else plural

    @staticmethod
    def _has_cue(statement: str, cues: List[str]) -> bool:
        normalized = VisualResponderImpl._normalize(statement)
        return any(re.search(rf"\b{re.escape(cue)}\b", normalized) for cue in cues)

    @staticmethod
    def _normalize(statement: str) -> str:
        # Removes straight and typographic apostrophes, so "what's", "what’s" and "whats" match alike
        return statement.strip().lower().replace("'", "").replace("\u2019", "").replace("\u2018", "")

    @staticmethod
    def _insert_a_an(label: str) -> str:
        article = "an" if label[:1].lower() in "aeiou" else "a"
        return f"{article} {label}"

#    def _point_to_objects(self, app, obj):
#        app.say("I can see {}".format(self._insert_a_an(obj.name)))
#        app.motion.point(obj.direction, speed=0.2)
#        app.motion.look(obj.direction, speed=0.1)
#        app.say("There it is!!")
