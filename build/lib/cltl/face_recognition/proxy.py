import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import jsonpickle
import logging
import numpy as np
import pickle
import requests
from cltl.backend.api.camera import Bounds
from emissor.representation.entity import Gender
from typing import Iterable, Tuple

from cltl.face_recognition.api import Face, FaceDetector
from cltl.combot.infra.docker import DockerInfra

logger = logging.getLogger(__name__)


FaceInfo = namedtuple('FaceInfo', ('gender',
                                   'age',
                                   'bbox',
                                   'embedding'))


class FaceDetectorProxy(FaceDetector):
    def __init__(self, start_infra: bool = True, detector_url: str = None, age_gender_url: str = None):
        if start_infra:
            self.detect_infra = DockerInfra('tae898/face-detection-recognition', 10002, 10002, False, 15)
            self.age_gender_infra = DockerInfra('tae898/age-gender', 10003, 10003, False, 15)
            self._detector_url = "http://127.0.0.1:10002/"
            self._age_gender_url = "http://127.0.0.1:10003/"
        else:
            self.detect_infra = None
            self.age_gender_infra = None
            self._detector_url = detector_url
            self._age_gender_url = age_gender_url

        if not self._detector_url or not self._age_gender_url:
            raise ValueError("No url defined for docker images: " + self._detector_url + ", " + self._age_gender_url)

    def __enter__(self):
        if self.detect_infra:
            executor = ThreadPoolExecutor(max_workers=2)
            detect = executor.submit(self.detect_infra.__enter__)
            age_gender = executor.submit(self.age_gender_infra.__enter__)

            detect.result()
            age_gender.result()
            executor.shutdown()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.detect_infra:
            executor = ThreadPoolExecutor(max_workers=2)
            detect = executor.submit(lambda: self.detect_infra.__exit__(exc_type, exc_val, exc_tb))
            age_gender = executor.submit(lambda: self.age_gender_infra.__exit__(exc_type, exc_val, exc_tb))

            detect.result()
            age_gender.result()
            executor.shutdown()

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Face], Iterable[Bounds]]:
        logger.info("Processing image %s", image.shape)

        face_infos = self.detect_faces(image)
        if face_infos:
            faces, bounds = zip(*map(self._to_face, face_infos))
        else:
            faces, bounds = (), ()

        return faces, bounds

    def _to_face(self, face_info: FaceInfo) -> Tuple[Face, Bounds]:
        bbox = [int(num) for num in face_info.bbox.tolist()]
        representation = face_info.embedding
        gender = Gender.MALE if face_info.gender["m"] > 0.5 else Gender.FEMALE
        age = round(face_info.age["mean"])

        return Face(representation, gender, age), Bounds(bbox[0], bbox[2], bbox[1], bbox[3])

    def to_binary_image(self, image: np.ndarray) -> bytes:
        is_success, buffer = cv2.imencode(".png", image)

        if not is_success:
            raise ValueError("Could not encode image")

        return buffer

    def run_face_api(self, to_send: dict) -> tuple:
        logger.debug(f"sending image to server...")

        start = time.time()
        to_send = jsonpickle.encode(to_send)
        response = requests.post(self._detector_url, json=to_send)
        logger.info("got %s from server in %s sec", response, time.time()-start)

        response = jsonpickle.decode(response.text)

        face_detection_recognition = response["face_detection_recognition"]
        logger.info(f"{len(face_detection_recognition)} faces deteced!")

        face_bboxes = [fdr["bbox"] for fdr in face_detection_recognition]
        det_scores = [fdr["det_score"] for fdr in face_detection_recognition]
        landmarks = [fdr["landmark"] for fdr in face_detection_recognition]

        embeddings = [fdr["normed_embedding"] for fdr in face_detection_recognition]

        return face_bboxes, det_scores, landmarks, embeddings


    def run_age_gender_api(self, embeddings: list) -> tuple:
        # -1 accounts for the batch size.
        data = np.array(embeddings).reshape(-1, 512).astype(np.float32)
        data = pickle.dumps(data)

        data = {"embeddings": data}
        data = jsonpickle.encode(data)
        logger.debug(f"sending embeddings to server ...")
        start = time.time()
        response = requests.post(self._age_gender_url, json=data)
        logger.info("got %s from server in %s sec", response, time.time()-start)

        response = jsonpickle.decode(response.text)
        ages = response["ages"]
        genders = response["genders"]

        return ages, genders

    def detect_faces(self, image: np.ndarray) -> Tuple[FaceInfo]:
        face_bboxes, det_scores, landmarks, embeddings = self.run_face_api({"image": self.to_binary_image(image)})

        ages, genders = self.run_age_gender_api(embeddings)

        return tuple(FaceInfo(*info) for info in zip(genders, ages, face_bboxes, embeddings))
