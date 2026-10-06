from datetime import datetime
from typing import Iterable, Optional, Sequence

from cltl.commons.discrete import UtteranceType

WORLD_NAMESPACE = "http://cltl.nl/leolani/world/"
INPUTS_NAMESPACE = "http://cltl.nl/leolani/inputs/"
SEM_HAS_ACTOR = "http://semanticweb.cs.vu.nl/2009/11/sem/hasActor"
SEM_HAS_TIME = "http://semanticweb.cs.vu.nl/2009/11/sem/hasTime"


def _to_uri_part(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def get_scene_id(scene: str, scenario_id: str) -> str:
    """Each scene type is a unique scene within the interaction (scenario)."""
    return f"{_to_uri_part(scene)}_{scenario_id}"


def get_context_capsule(scenario_id: str, place: Optional[str] = None, place_id: Optional[str] = None,
                        country: str = "unknown", region: str = "unknown", city: str = "unknown"):
    """Context of the interaction, see LongTermMemory.capsule_context."""
    return {
        "context_id": scenario_id,
        "date": datetime.now().date(),
        "place": place,
        "place_id": place_id,
        "country": country,
        "region": region,
        "city": city,
    }


def get_capsule_from_scene(scene: str, scene_id: str, objects: Iterable[str], image_id: str, scenario_id: str,
                           source: str, confidence: float, region: Sequence[int]):
    """
    Capsule with event_details for LongTermMemory.capsule_event: the scene is a situation that
    has the objects in it as actors.

    Parameters
    ----------
    objects : Iterable[str]
        The labels of the objects seen in the scene.
    region : Sequence[int]
        The bounds of the image (x0, y0, x1, y1), the scene annotates the whole image.
    """
    capsule = {
        "visual": scenario_id,
        "detection": image_id,
        "source": {"label": source, "type": ["sensor"], 'uri': INPUTS_NAMESPACE + _to_uri_part(source)},
        "image": image_id,
        "region": list(region),
        "utterance_type": UtteranceType.EXPERIENCE,
        'confidence': confidence,
        # As LongTermMemory.capsule_experience does for detections: the confidence is the certainty
        # (CERTAIN above 0.9, PROBABLE from 0.5, POSSIBLE above 0) and what is seen is positive
        "perspective": {"certainty": confidence, "polarity": 1},
        "timestamp": datetime.now(),
        "context_id": scenario_id
    }
    event_details = []
    subject = {"label": scene, "type": ["situation"], "uri": WORLD_NAMESPACE + scene_id}
    time = capsule["timestamp"].isoformat(timespec="minutes")
    predicate = {"label": "hasTime", "uri": SEM_HAS_TIME}
    object = {"label": time, "type": ["date"], "uri": WORLD_NAMESPACE + time.replace(":", "-")}
    event_details.append({"subject": subject, "predicate": predicate, "object": object})
    for label in objects:
        object = {"label": label, "type": ["object"], "uri": WORLD_NAMESPACE + _to_uri_part(label)}
        predicate = {"label": "hasActor", "uri": SEM_HAS_ACTOR}
        triple = {"subject": subject, "predicate": predicate, "object": object}
        event_details.append(triple)

    capsule["event_details"] = event_details
    return capsule
