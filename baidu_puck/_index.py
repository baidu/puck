"""Shared high-level wrapper around the SWIG extension."""

from typing import Any, Tuple

import numpy as np

try:
    from . import py_puck_api as _native
except ImportError:
    import py_puck_api as _native


class Index:
    """Common lifecycle and search API for all index implementations."""

    def __init__(self, index_type: int, **config: Any) -> None:
        self._loaded = False
        self._config = dict(config)
        self._configure_gflags()
        self._native = _native.PySearcher(index_type)

    def _configure_gflags(self) -> None:
        for name, value in self._config.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            if not _native.update_gflag(str(name), str(value)):
                raise ValueError(f"invalid configuration: {name}={value}")

    @property
    def dim(self) -> int:
        return self._native.dim()

    def train(self) -> "Index":
        """Train and persist the index codebooks, then return this index."""
        self._configure_gflags()
        self._check(self._native.train(), "train")
        self._loaded = False
        return self

    def build(self) -> "Index":
        """Build and persist the index, then return this index."""
        self._configure_gflags()
        self._check(self._native.build(), "build")
        self._loaded = False
        return self

    def load(self) -> "Index":
        """Load the persisted index into searchable memory and return this index."""
        self._configure_gflags()
        self._check(self._native.load(), "load")
        self._loaded = True
        return self

    def search(
        self,
        queries: np.ndarray,
        top: int,
        *,
        ef_search: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search loaded index and return distance and label matrices."""
        if not self._loaded:
            raise RuntimeError("search requires load() to be called first")
        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2:
            raise ValueError("queries must be a one- or two-dimensional array")
        if self.dim <= 0 or queries.shape[1] != self.dim:
            raise ValueError(
                f"query dimension {queries.shape[1]} does not match index dimension {self.dim}"
            )
        if top <= 0:
            raise ValueError("top must be positive")
        if ef_search < 0:
            raise ValueError("ef_search must be non-negative")

        distances = np.full((queries.shape[0], top), np.inf, dtype=np.float32)
        labels = np.full(
            (queries.shape[0], top), np.iinfo(np.uint32).max, dtype=np.uint32
        )
        result = self._native.search(
            queries.shape[0],
            _native.swig_ptr(queries),
            top,
            ef_search,
            _native.swig_ptr(distances),
            _native.swig_ptr(labels),
        )
        self._check(result, "search")
        return distances, labels

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with status {result}")
