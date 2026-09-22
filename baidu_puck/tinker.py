"""Tinker graph-index implementation."""

from typing import Any, Tuple

import numpy as np

from ._index import Index as _Index

INDEX_TYPE = 2


class Tinker(_Index):
    """Tinker index configured once during construction."""

    def __init__(self, **config: Any) -> None:
        super().__init__(INDEX_TYPE, **config)

    def search(
        self,
        queries: np.ndarray,
        top: int,
        *,
        ef_search: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search queries and return distance and label matrices.

        Args:
            queries: One vector or a batch matching the index dimension.
            top: Positive maximum result count per query.
            ef_search: Optional non-negative graph width. Defaults to 0, which
                uses top; positive values use max(ef_search, top).

        Returns:
            A tuple of distance and label matrices.

        Raises:
            ValueError: Invalid dimensions, top, or negative ef_search.
            RuntimeError: Index is not loaded or native search fails.
        """
        if ef_search < 0:
            raise ValueError("ef_search must be non-negative")
        return super().search(queries, top, ef_search=ef_search)
