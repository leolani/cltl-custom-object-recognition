import uuid

from dataclasses import dataclass
from typing import Iterable, List

from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.event.emissor import AnnotationEvent
from emissor.representation.container import Ruler
from emissor.representation.scenario import Mention, Annotation

from cltl.vector_id.api import VectorIdentity


@dataclass
class VectorIdentityEvent(AnnotationEvent[Annotation[str]]):
    @classmethod
    def create(cls, segments: Iterable[List[Ruler]], ids: Iterable[str],
               source: str = __name__, timestamp: int = None):
        def to_mention(seg_id):
            ts = timestamp if timestamp is not None else timestamp_now()
            annotation = Annotation(VectorIdentity.__name__, seg_id[1], source, ts)

            return Mention(str(uuid.uuid4()), [seg_id[0]], [annotation])

        seg_ids = zip(segments, ids)

        return cls(cls.__name__, list(map(to_mention, seg_ids)))