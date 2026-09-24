import logging
import os.path
import pickle
import uuid
from typing import List

import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from cltl.vector_id.api import VectorIdentity

logger = logging.getLogger(__name__)


_REPRESENTATIONS = "representations.pkl"
_CENTROIDS = "centroids.pkl"


class ClusterIdentity(VectorIdentity):
    @classmethod
    def agglomerative(cls, ndim: int, distance_threshold: float, storage_path: str = None):
        ac = AgglomerativeClustering(n_clusters=None,
                                     metric='cosine',
                                     linkage='average',
                                     distance_threshold=distance_threshold)

        return cls(ac, ndim, storage_path)

    def __init__(self, clustering, ndim: int, storage_path: str = None):
        self._clustering = clustering
        self._storage_path = storage_path

        self._centroids = self._load(_CENTROIDS, dict())
        self._representations = self._load(_REPRESENTATIONS, np.empty((0, ndim)) if ndim else None)

    def _load(self, path, default):
        if not self._storage_path:
            logger.info("Running in-memory only for %s", path)
            return default

        os.makedirs(self._storage_path, exist_ok=True)
        full_path = os.path.join(self._storage_path, path)

        if os.path.isfile(full_path):
            with open(full_path, 'rb') as infile:
                data = pickle.load(infile)

            logger.info("Loaded %s with %s elements", path, len(data) if data is not None else None)

            return data

        logger.info("Initialized %s with default %s", path, default)

        return default

    def _dump(self, obj, storage, path):
        if not self._storage_path:
            return

        full_path = os.path.join(storage, path)

        with open(full_path, 'wb') as outfile:
            pickle.dump(obj, outfile)

    def add(self, representations: np.ndarray) -> List[str]:
        start = time.time()

        if self._representations is None:
            self._representations = np.atleast_2d(representations)
        else:
            self._representations = np.vstack([self._representations, representations])

        if len(self._representations) == 1:
            id = str(uuid.uuid4())
            self._centroids = {id: normalize(self._representations)[0]}

            return [id]

        cluster_ids = self._cluster_representations(self._centroids, self._representations)

        id_map = {id: idx for idx, id in enumerate(set(cluster_ids))}
        labeled_representations = np.hstack([np.atleast_2d([id_map[id] for id in cluster_ids]).T,
                                             normalize(self._representations)])
        self._centroids = {id: self._centroid(labeled_representations, label) for id, label in id_map.items()}

        self._dump(self._centroids, self._storage_path, _CENTROIDS)
        self._dump(self._representations, self._storage_path, _REPRESENTATIONS)

        if time.time() - start > 1:
            logger.warning("Clustering took %s seconds", time.time() - start)

        return cluster_ids[-len(representations):]

    def _centroid(self, labeled_representations: np.array, label: int):
        mask = (labeled_representations[:, 0] == label)

        return labeled_representations[mask][:, 1:].mean(axis=0)

    def _cluster_representations(self, centroids, representations):
        logger.debug("Clustering ids")

        if len(representations) == 0:
            return []

        if len(centroids):
            ids, id_representations = zip(*centroids.items())
            all_ = np.vstack([id_representations, representations])
        else:
            ids, id_representations, all_ = (), (), representations

        clustering = self._clustering.fit(all_)

        label_ids = {label: id for id, label in zip(ids, clustering.labels_[:len(ids)])}
        label_ids = {label: label_ids[label] if label in label_ids else str(uuid.uuid4())
                     for label in np.unique(clustering.labels_)}

        return [label_ids[label] for label in clustering.labels_[len(ids):]]


