"""Puck index implementation."""

from typing import Any

from ._index import Index as _Index

INDEX_TYPE = 1


class Puck(_Index):
    """Puck index configured once during construction."""

    def __init__(self, **config: Any) -> None:
        super().__init__(INDEX_TYPE, **config)
