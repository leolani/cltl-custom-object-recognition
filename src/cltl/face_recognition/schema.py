import uuid
from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, ImageSignal, Annotation
from typing import Iterable

from cltl.face_recognition.api import Face


@dataclass
class FaceRecognitionEvent(AnnotationEvent[Annotation[Face]]):
    @classmethod
    def create(cls, image_signal: ImageSignal, faces: Iterable[Face], bounds: Iterable[Bounds]):
        if faces:
            mentions = [FaceRecognitionEvent.to_mention(image_signal, face, bound)
                        for face, bound in zip(faces, bounds)]
        else:
            mentions = [FaceRecognitionEvent.to_mention(image_signal)]

        return cls(cls.__name__, mentions)

    @staticmethod
    def to_mention(image_signal: ImageSignal, face: Face = None, bounds: Bounds = None):
        """
        Create Mention with face annotations. If no face is detected, annotate the whole
        image with Face Annotation with value None.
        """
        segment = image_signal.ruler
        if bounds:
            clipped = Bounds(segment.bounds[0], segment.bounds[2],
                             segment.bounds[1], segment.bounds[3]).intersection(bounds)
            segment = segment.get_area_bounding_box(clipped.x0, clipped.y0, clipped.x1, clipped.y1)

        annotation = Annotation(Face.__name__, face, __name__, timestamp_now())

        return Mention(str(uuid.uuid4()), [segment], [annotation])