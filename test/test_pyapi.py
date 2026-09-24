import importlib
import sys
import types

import numpy as np
import pytest


class FakeSearcher:
    def __init__(self, index_type):
        self._index_type = index_type
        self.calls = []

    def dim(self):
        return 4

    def train(self):
        self.calls.append("train")
        return 0

    def build(self):
        self.calls.append("build")
        return 0

    def load(self):
        self.calls.append("load")
        return 0

    def search(self, n, queries, topk, ef_search, distances, labels):
        self.calls.append(("search", n, topk, ef_search))
        distances[:] = np.arange(n * topk, dtype=np.float32).reshape(n, topk)
        labels[:] = np.arange(n * topk, dtype=np.uint32).reshape(n, topk)
        return 0


@pytest.fixture()
def api(monkeypatch):
    configured = {}
    native = types.ModuleType("baidu_puck.py_puck_api")
    native.PySearcher = lambda index_type: FakeSearcher(index_type)
    def update_gflag(key, value):
        configured[key] = value
        return True

    native.update_gflag = update_gflag
    native.swig_ptr = lambda value: value
    native.configured = configured
    monkeypatch.setitem(sys.modules, "baidu_puck.py_puck_api", native)
    for name in list(sys.modules):
        if name == "baidu_puck" or name.startswith("baidu_puck.") and name != "baidu_puck.py_puck_api":
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("baidu_puck")


@pytest.mark.parametrize(
    ("factory_name", "index_type"),
    [("hierarchical_cluster", 0), ("puck", 1), ("tinker", 2)],
)
def test_uniform_lifecycle(api, factory_name, index_type):
    index = getattr(api, factory_name)(feature_dim=4)
    assert index._native._index_type == index_type
    assert api.py_puck_api.configured["feature_dim"] == "4"
    assert index.build() is index
    assert index.load() is index
    assert index._native.calls == ["build", "load"]


def test_numpy_search(api):
    index = api.tinker(feature_dim=4).load()
    distances, labels = index.search(np.ones((2, 4), dtype=np.float64), top=3, ef_search=40)
    assert distances.shape == (2, 3)
    assert labels.dtype == np.uint32
    assert index._native.calls[-1] == ("search", 2, 3, 40)


def test_search_rejects_wrong_dimension(api):
    with pytest.raises(ValueError, match="does not match"):
        api.puck().load().search(np.ones((2, 3), dtype=np.float32), top=2)


def test_search_requires_load(api):
    with pytest.raises(RuntimeError, match="load"):
        api.puck(feature_dim=4).search(np.ones((1, 4), dtype=np.float32), top=1)


def test_tinker_defaults_to_zero_and_keeps_ef_search_keyword_only(api):
    """Omitted and explicit zero widths reach native; positional widths fail."""
    index = api.tinker(feature_dim=4).load()
    queries = np.ones((1, 4), dtype=np.float32)
    distances, labels = index.search(queries, top=3)
    assert distances.shape == labels.shape == (1, 3)
    assert index._native.calls[-1] == ("search", 1, 3, 0)
    index.search(queries, top=3, ef_search=0)
    assert index._native.calls[-1] == ("search", 1, 3, 0)
    with pytest.raises(TypeError):
        index.search(queries, 3, 10)
    assert index._native.calls == ["load", ("search", 1, 3, 0), ("search", 1, 3, 0)]


@pytest.mark.parametrize("ef_search", [-1, -(2 ** 32), -(2 ** 63)])
def test_tinker_rejects_negative_width(api, ef_search):
    """Invalid widths never reach the native extension."""
    index = api.tinker(feature_dim=4).load()
    with pytest.raises(ValueError, match="ef_search must be non-negative"):
        index.search(np.ones((1, 4)), top=3, ef_search=ef_search)
    assert index._native.calls == ["load"]


def test_tinker_accepts_width_below_top(api):
    """Pass a positive narrow width to native, where promotion takes place."""
    index = api.tinker(feature_dim=4).load()
    distances, labels = index.search(np.ones((1, 4)), top=3, ef_search=1)
    assert distances.shape == labels.shape == (1, 3)
    assert index._native.calls[-1] == ("search", 1, 3, 1)


@pytest.mark.parametrize("factory_name", ["hierarchical_cluster", "puck"])
def test_other_engines_keep_search_contract(api, factory_name):
    """HC/Puck still accept omitted/zero/positive ef_search and reject negatives."""
    index = getattr(api, factory_name)(feature_dim=4).load()
    queries = np.ones((1, 4))
    index.search(queries, top=3)
    assert index._native.calls[-1] == ("search", 1, 3, 0)
    for width in (0, 1, 40):
        index.search(queries, top=3, ef_search=width)
        assert index._native.calls[-1] == ("search", 1, 3, width)
    with pytest.raises(ValueError, match="non-negative"):
        index.search(queries, top=3, ef_search=-1)
