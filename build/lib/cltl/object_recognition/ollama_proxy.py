import json
import logging
import re
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from cltl.backend.api.camera import Bounds
from ollama import Client

from cltl.object_recognition.api import Object, ObjectDetector

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "qwen2.5vl"

# Object.type used for the whole-image scene classification, as opposed to individual
# object detections, which use the model name (see OllamaObjectDetectorProxy._to_object).
_SCENE_TYPE = "scene"

_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "string",
            "description": "A short label classifying the overall scene or place depicted in the "
                            "image, e.g. 'office', 'kitchen', 'living room', 'street', 'city', "
                            "'village', 'forest'.",
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["label", "box_2d"],
            },
        }
    },
    "required": ["scene", "objects"],
}

_PROMPT = (
    "Analyze this image. First classify the overall scene or place it depicts with a short "
    "label, e.g. \"office\", \"kitchen\", \"living room\", \"street\", \"city\", \"village\", "
    "\"forest\", and report it as \"scene\". "
    "Then detect every distinct object in the image. For each object report its label "
    "and a bounding box as \"box_2d\": [ymin, xmin, ymax, xmax], with coordinates "
    "normalized to the range 0-1000 relative to the image height and width. "
    "If you can estimate your confidence, add a \"confidence\" value between 0 and 1. "
    "Report only objects you can actually see."
)


class OllamaObjectDetectorProxy(ObjectDetector):
    """
    ObjectDetector implementation that uses a vision-language model (e.g. Qwen2.5-VL)
    served through Ollama -- either a local instance or Ollama's cloud API
    (https://ollama.com) -- instead of a dedicated object detection model like Yolo5.

    The model is prompted to return detections with normalized bounding boxes as
    structured JSON, which is parsed into the same Object/Bounds shape produced by
    ObjectDetectorProxy, so this can be used as a drop-in alternative.

    In addition to individual objects, the model is asked to classify the overall
    scene depicted (e.g. "office", "kitchen", "street"). This is returned as an
    additional Object with type _SCENE_TYPE, whose Bounds cover the complete image.
    """

    @classmethod
    def from_config(cls, config_manager):
        config = config_manager.get_config("cltl.object_recognition.ollama")

        return cls(model=config.get("model") if "model" in config else _DEFAULT_MODEL,
                    host=config.get("host") if "host" in config else None,
                    api_key=config.get("api_key") if "api_key" in config else None)

    def __init__(self, model: str = _DEFAULT_MODEL, host: str = None, api_key: str = None):
        """
        Parameters
        ----------
        model : str
            Name of the (vision-capable) Ollama model to use, e.g. "qwen2.5vl".
        host : str
            Address of the Ollama server. If not given, falls back to the
            OLLAMA_HOST environment variable, or a local Ollama instance.
            Set to "https://ollama.com" to use Ollama's cloud API.
        api_key : str
            API key for Ollama's cloud API. If not given, falls back to the
            OLLAMA_API_KEY environment variable. Not needed for a local instance.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = Client(host=host, headers=headers)
        self._model = model

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Object], Iterable[Bounds]]:
        logger.info("Processing image %s with model %s", image.shape, self._model)

        height, width = image.shape[:2]
        result = self._detect(image)

        objects = []
        bounds = []

        scene = result.get("scene")
        if scene:
            objects.append(Object(_SCENE_TYPE, scene, None))
            bounds.append(Bounds(0, width, 0, height))

        for detection in result.get("objects", ()):
            parsed = self._to_object(detection, width, height)
            if parsed is None:
                continue
            obj, bound = parsed
            objects.append(obj)
            bounds.append(bound)

        logger.info("Detected scene '%s' and objects: %s", scene, [obj.label for obj in objects if obj.type != _SCENE_TYPE])

        return tuple(objects), tuple(bounds)

    def _detect(self, image: np.ndarray) -> dict:
        response = self._client.chat(
            model=self._model,
            messages=[{
                "role": "user",
                "content": _PROMPT,
                "images": [self._to_binary_image(image)],
            }],
            format=_DETECTION_SCHEMA,
            # Reasoning/"thinking" models (e.g. Qwen3-VL) put their reasoning in a
            # separate `message.thinking` field when this is respected, keeping
            # `message.content` limited to the actual (JSON) answer.
            think=False,
        )

        content = response["message"]["content"]
        return self._parse_response(content)

    def _parse_response(self, content: str) -> dict:
        # Some models (e.g. qwen3-vl:8b, see ollama/ollama#14798) ignore think=False
        # and emit <think>...</think> reasoning directly into the content instead of
        # the separate `thinking` field, so strip it out before parsing as JSON.
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # As a last resort, look for the first top-level JSON object in the text,
        # in case the model added commentary around it despite the schema constraint.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse detection response as JSON: %s", content)
        return {}

    def _to_object(self, detection: dict, width: int, height: int) -> Optional[Tuple[Object, Bounds]]:
        label = detection.get("label")
        box = detection.get("box_2d")
        if not label or not box or len(box) != 4:
            logger.warning("Skipping malformed detection: %s", detection)
            return None

        ymin, xmin, ymax, xmax = box
        x0 = max(0.0, min(width, xmin / 1000 * width))
        x1 = max(0.0, min(width, xmax / 1000 * width))
        y0 = max(0.0, min(height, ymin / 1000 * height))
        y1 = max(0.0, min(height, ymax / 1000 * height))
        if x1 <= x0 or y1 <= y0:
            logger.warning("Skipping detection with empty bounding box: %s", detection)
            return None

        return Object(self._model, label, detection.get("confidence")), Bounds(x0, x1, y0, y1)

    def _to_binary_image(self, image: np.ndarray) -> bytes:
        is_success, buffer = cv2.imencode(".png", image)

        if not is_success:
            raise ValueError("Could not encode image")

        return bytes(buffer)

# python -m cltl.object_recognition.ollama_proxy  <image_path> [--model ...] [--host ...] [--api-key ...] [--show]
# python -m cltl.object_recognition.ollama_proxy data/image-1.jpeg --model qwen3-vl:2b-instruct
def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Detect objects in an image using a VLM served through Ollama.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--host", default=None, help="Ollama host, e.g. https://ollama.com for the cloud API")
    parser.add_argument("--api-key", default=None, help="Ollama API key (falls back to OLLAMA_API_KEY env var)")
    parser.add_argument("--output", default=None,
                         help="Where to save the annotated image (default: <image>.detections.png)")
    parser.add_argument("--show", action="store_true",
                         help="Also open the annotated image in a window (blocks until a key is pressed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    proxy = OllamaObjectDetectorProxy(model=args.model, host=args.host, api_key=args.api_key)
    objects, bounds = proxy.detect(image)

    found_objects = False
    for obj, bound in zip(objects, bounds):
        if obj.type == _SCENE_TYPE:
            print(f"Scene: {obj.label}")
            cv2.putText(image, f"scene: {obj.label}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            continue

        found_objects = True
        print(f"{obj.label} (confidence={obj.confidence}): {bound}")
        cv2.rectangle(image, (int(bound.x0), int(bound.y0)), (int(bound.x1), int(bound.y1)), (0, 255, 0), 2)
        cv2.putText(image, obj.label, (int(bound.x0), max(0, int(bound.y0) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if not found_objects:
        print("No objects detected")

    output_path = args.output or f"{os.path.splitext(args.image)[0]}.detections.png"
    cv2.imwrite(output_path, image)
    print(f"Annotated image written to {output_path}")

    if args.show:
        cv2.imshow("Detections", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
