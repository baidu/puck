# Python API Reference

## Import

```python
from baidu_puck import hierarchical_cluster, puck, tinker
```

| Constructor | Index type |
| --- | --- |
| `hierarchical_cluster(**config)` | Hierarchical Cluster |
| `puck(**config)` | Puck |
| `tinker(**config)` | Tinker |

Configuration is accepted only by the constructor. Lifecycle methods do not accept configuration arguments.

## Lifecycle

Build and load an index:

```python
index = puck(**config)
index.train()
index.build()
index.load()
distances, labels = index.search(queries, top=10)
```

Load an existing index:

```python
index = puck(index_path="puck_index")
index.load()
distances, labels = index.search(queries, top=10)
```

### `train()`

- Arguments: none.
- Returns: the current index object.
- Behavior: reads `index_path/feature_file_name`, trains codebooks or a quantizer, and writes the training artifacts.
- Raises: `RuntimeError` on failure.

### `build()`

- Arguments: none.
- Returns: the current index object.
- Behavior: reads all vectors, builds the index, and writes the index files.
- Precondition: `train()` has completed, or compatible training artifacts exist in `index_path`.
- Raises: `RuntimeError` on failure.

`train()` and `build()` write their outputs synchronously. There is no `save()` method.

### `load()`

- Arguments: none.
- Returns: the current index object.
- Behavior: loads the persisted index into memory.
- Raises: `RuntimeError` if an index file is missing, invalid, incomplete, or incompatible.

`search()` runs only after `load()`. `build()` writes the index but does not load it.

### `dim`

Read-only `int` property containing the query dimension. After `load()`, the value comes from persisted model metadata.

## Common Configuration

The following constructor arguments are accepted by all index types.

| Argument | Type | Default | Used by | Description |
| --- | --- | --- | --- | --- |
| `index_path` | `str` | `"puck_index"` | train/build/load | Input and output directory. |
| `index_file_name` | `str` | `"index.dat"` | train/build/load | Model metadata file name. |
| `feature_file_name` | `str` | `"all_data.feat.bin"` | train/build | Input fvecs file name. |
| `feature_dim` | `int` | `256` | train/build | Input vector dimension. After load, the persisted value is used. |
| `whether_norm` | `bool` | `True` | train/build/search | Enables vector and query normalization. |
| `threads_count` | `int` | hardware concurrency | train/build | Training and build thread count. |
| `context_initial_pool_size` | `int` | hardware concurrency | search | Batch-search thread count and initial search-context pool size. |
| `label_file_name` | `str` | `"all_data.url"` | build/load | Label file name. |

Unknown argument names or invalid gflags values raise `ValueError`.

Configuration uses process-wide gflags. Concurrent operations on indexes with different configurations are not supported.

## Hierarchical Cluster

### Constructor

```python
index = hierarchical_cluster(
    index_path="hc_index",
    feature_file_name="all_data.feat.bin",
    feature_dim=128,
    coarse_cluster_count=100,
    fine_cluster_count=100,
    search_coarse_count=20,
    neighbors_count=40000,
    topk=100,
)
```

Accepts common configuration and the arguments below.

| Argument | Type | Default | Used by | Description |
| --- | --- | --- | --- | --- |
| `coarse_cluster_count` | `int` | `2000` | train/build/load/search | Number of coarse centroids. |
| `fine_cluster_count` | `int` | `2000` | train/build/load/search | Number of fine centroids per coarse cluster. |
| `coarse_codebook_file_name` | `str` | `"coarse.dat"` | train/build/load | Coarse codebook file name. |
| `fine_codebook_file_name` | `str` | `"fine.dat"` | train/build/load | Fine codebook file name. |
| `cell_assign_file_name` | `str` | `"cell_assign.dat"` | build/load | Point-to-cell assignment file name. |
| `train_points_count` | `int` | `5000000` | train | Maximum cluster-training sample count. |
| `train_fea_file_name` | `str` | `"mid-data/train_clusters.dat"` | train | Temporary cluster-training data file. |
| `kmeans_init_berkeley` | `bool` | `True` | train | Enables Berkeley KMeans initialization. |
| `kmeans_iterations_count` | `int` | `10` | train | KMeans iteration count. |
| `thread_chunk_size` | `int` | `10000` | train/build | Parallel processing chunk size. |
| `search_coarse_count` | `int` | `200` | load/search | Number of coarse clusters searched, capped by `coarse_cluster_count`. |
| `neighbors_count` | `int` | `40000` | load/search | Maximum number of candidate points scanned; the minimum value is `topk`. |
| `topk` | `int` | `100` | load/search | Maximum accepted value of `search(top=...)`. |
| `ip2cos` | `int` | `0` | train/build/load/search | IP-to-cosine mode: `0` disabled, `1` direct, `2` pre-transformed. |

### Methods

| Method | Arguments | Returns | Description |
| --- | --- | --- | --- |
| `train()` | none | current object | Trains and writes two-level cluster codebooks. |
| `build()` | none | current object | Builds and writes assignments and feature data. |
| `load()` | none | current object | Loads the index. |
| `search(queries, top)` | `queries`, `top` | `(distances, labels)` | Searches the index. |

### `search(queries, top)`

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `queries` | array-like | yes | One vector with shape `(D,)` or a batch with shape `(N, D)`. |
| `top` | `int` | yes | Maximum result count per query. The accepted range is `1` through `topk`. |

Search scope is controlled by `search_coarse_count` and `neighbors_count`.

## Puck

### Constructor

```python
index = puck(
    index_path="puck_index",
    feature_file_name="all_data.feat.bin",
    feature_dim=128,
    coarse_cluster_count=100,
    fine_cluster_count=100,
    filter_nsq=32,
    nsq=128,
    filter_topk=110,
    topk=100,
)
```

Accepts common and Hierarchical Cluster configuration, plus the arguments below.

| Argument | Type | Default | Used by | Description |
| --- | --- | --- | --- | --- |
| `whether_pq` | `bool` | `True` | train/build/load/search | Enables PQ encoding. |
| `nsq` | `int` | `feature_dim` | train/build/load/search | Number of PQ subspaces. |
| `pq_train_points_count` | `int` | `1000000` | train | Maximum PQ-training sample count. |
| `train_pq_file_name` | `str` | `"mid-data/train_pq.dat"` | train | Temporary PQ-training data file. |
| `pq_codebook_file_name` | `str` | `"learn_codebooks.dat"` | train/build/load | PQ codebook file name. |
| `pq_data_file_name` | `str` | `"pq_data.dat"` | build/load/search | PQ-encoded feature file name. |
| `filter_nsq` | `int` | `feature_dim / 4` | train/build/load/search | Number of filter-quantization subspaces. |
| `filter_codebook_file_name` | `str` | `"filter_codebook.dat"` | train/build/load | Filter codebook file name. |
| `filter_data_file_name` | `str` | `"filter_data.dat"` | build/load/search | Filter-encoded feature file name. |
| `filter_topk` | `int` | `topk * 11` | load/search | Candidate count retained after filtering. |
| `radius_rate` | `float` | `1.0` | load/search | Candidate filtering radius multiplier. |

### Methods

| Method | Arguments | Returns | Description |
| --- | --- | --- | --- |
| `train()` | none | current object | Trains and writes cluster, filter, and PQ codebooks. |
| `build()` | none | current object | Builds and writes assignments and quantized features. |
| `load()` | none | current object | Loads the index. |
| `search(queries, top)` | `queries`, `top` | `(distances, labels)` | Searches the index. |

### `search(queries, top)`

| Argument | Type | Required | Description |
| --- | --- | --- | --- |
| `queries` | array-like | yes | One vector with shape `(D,)` or a batch with shape `(N, D)`. |
| `top` | `int` | yes | Maximum result count per query. The accepted range is `1` through `topk`. |

Search scope is controlled by `search_coarse_count`, `filter_topk`, and `radius_rate`.

## Tinker

### Constructor

```python
index = tinker(
    index_path="tinker_index",
    feature_file_name="all_data.feat.bin",
    feature_dim=128,
    whether_norm=True,
    tinker_quantizer_type="LVQ8x8",
    tinker_neighborhood_count=16,
    tinker_ef_construction=600,
    tinker_memory_capacity=4 * 1024**3,
)
```

Accepts common configuration and the arguments below.

| Argument | Type | Default | Used by | Description |
| --- | --- | --- | --- | --- |
| `tinker_quantizer_type` | `str` | `"FLAT"` | train/build | One of `FLAT`, `LVQ8x8`, `LVQ8x4`, `LVQ8x0`, `LVQ4x8`, `LVQ4x4`, or `LVQ4x0`. |
| `tinker_neighborhood_count` | `int` | `16` | build/load/search | Maximum graph degree. |
| `tinker_ef_construction` | `int` | `600` | build | Build candidate width. |
| `tinker_memory_capacity` | `int` | `42949672960` | build/load | Maximum index memory in bytes. |
| `tinker_quantizer_file_name` | `str` | `"quantizer.dat"` | train/build/load | Quantizer file name. |
| `tinker_file_prefix` | `str` | `"tinker"` | build/load | Prefix for encoded-feature and graph files. |
| `tinker_idmapper_file_name` | `str` | `"idmapper.dat"` | build/load | Memory-node to result-ID mapping file. |
| `tinker_brief` | `bool` | `False` | build/load/search | Enables label-filter metadata. |
| `brief_file_name` | `str` | `"all_data.brief"` | build | Tinker brief input file under `index_path`, with one tab-separated filter-label list per vector. |
| `tinker_cartesian` | `bool` | `True` | build/load/search | Uses Cartesian label combinations in brief mode. |

### Methods

| Method | Arguments | Returns | Description |
| --- | --- | --- | --- |
| `train()` | none | current object | Trains and writes the FLAT/LVQ quantizer. |
| `build()` | none | current object | Builds and writes the graph index. |
| `load()` | none | current object | Loads the graph index. |
| `search(queries, top, *, ef_search=0)` | `queries`, `top`, optional `ef_search` | `(distances, labels)` | Searches the graph. |

### `search(queries, top, *, ef_search=0)`

| Argument | Type | Default | Required | Description |
| --- | --- | --- | --- | --- |
| `queries` | array-like | none | yes | One vector with shape `(D,)` or a batch with shape `(N, D)`. |
| `top` | `int` | none | yes | Maximum result count per query. Must be positive. |
| `ef_search` | `int` | `0` | no, keyword-only | Non-negative graph search width. Zero uses `top`; positive values use `max(ef_search, top)`. Negative values are rejected. |

`ef_search` may be omitted: `index.search(queries, top=10)` uses a graph width
of 10, as does `index.search(queries, top=10, ef_search=0)`. To request a wider
search, use `index.search(queries, top=10, ef_search=50)`. There is no global
search-width configuration.
HC/Puck search signatures and behavior are unchanged.

With `tinker_brief=True`, `label_file_name` holds one tab-separated label list per vector. Filtered search is not exposed by this API.

## Search Input and Output

`queries` is converted to a C-contiguous `numpy.float32` array.

- A single query has shape `(D,)` and is converted to `(1, D)`.
- A batch has shape `(N, D)`.
- `D` equals `index.dim`.

`search()` returns `(distances, labels)`:

- `distances`: `numpy.float32` with shape `(N, top)`.
- `labels`: `numpy.uint32` with shape `(N, top)`.

If fewer than `top` results are available, unused distances are `inf` and unused labels are `numpy.iinfo(numpy.uint32).max`.

Invalid dimensions, `top`, or `ef_search` raise `ValueError`. Calling `search()` before `load()`, or a native search failure, raises `RuntimeError`.

## Feature File Format

Training and build input uses little-endian fvecs records:

```text
int32 dimension + float32[dimension]
```

Each record occupies `4 + 4 * dimension` bytes. All records share the same dimension, equal to `feature_dim`.

## Runtime Constraints

- Configuration uses process-wide gflags. Concurrent operations on differently configured indexes in one process are not supported.
- `load()` determines the native index type from `index.dat`. The constructor type matches the persisted index.
- `_PyPuck.so` is tied to the target Python ABI, Linux ABI, and AVX2 instruction set.
