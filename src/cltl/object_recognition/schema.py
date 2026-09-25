import logging
import uuid
from dataclasses import dataclass
from typing import Iterable

from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from emissor.representation.scenario import Mention, ImageSignal, Annotation, module_source, class_type

from cltl.object_recognition.api import Object

logger = logging.getLogger(__name__)

@dataclass
class ObjectRecognitionEvent(AnnotationEvent[Annotation[Object]]):
    @classmethod
    def create_obj_rec_event(cls, image_signal: ImageSignal, objects: Iterable[Object], bounds: Iterable[Bounds]):
        if objects:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal, object, bound)
                        for object, bound in zip(objects, bounds)]
        else:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal)]

        return cls(cls.__name__, mentions)

    @staticmethod
    def to_mention(image_signal: ImageSignal, object: Object = None, bounds: Bounds = None):
        """
        Create Mention with object annotations. If no face is detected, annotate the whole
        image with Object Annotation with value None.
        """
        segment = image_signal.ruler
        if bounds:
            clipped = Bounds.from_diagonal(*segment.bounds).intersection(bounds)
            if clipped:
                segment = segment.get_area_bounding_box(clipped.x0, clipped.y0, clipped.x1, clipped.y1)
            else:
                # The signal's bounds don't match the image's pixel size (e.g. the
                # backend's (-1, -1) fallback), so annotate the whole image instead.
                logger.warning("Object bounds %s outside image signal bounds %s, annotating whole image",
                               bounds, segment.bounds)

        annotation = Annotation(class_type(object), object, module_source(__name__), timestamp_now())

        return Mention(str(uuid.uuid4()), [segment], [annotation])


if __name__ == '__main__':
    from emissor.representation.util import marshal
    signal = ImageSignal.for_scenario("sc_id1", 0, 1, "", (0,0,1,1))
    event = ObjectRecognitionEvent.create_obj_rec_event(signal, [Object("chair")], [Bounds(0,1,0,1)])
    print(marshal(event))