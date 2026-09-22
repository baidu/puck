#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

Reproducible, subprocess-isolated benchmark for the real baidu_puck API.
Maintainer: Puck contributors (contact: repository issue tracker).
See benchmark.md for measurement definitions and operational guidance.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid


SCHEMA = 1
ENGINES = ("hc", "puck", "tinker")
SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parent.parent


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _defaults(smoke=False):
    count = 4096 if smoke else 1000000
    return {
        "data": {"count": count, "queries": 32 if smoke else 10000, "dim": 32, "seed": 20260921, "k": 100},
        "gt": {"backend": "auto", "query_block": 32, "db_block": 16384, "threads": _cpu_count()},
        "runtime": {
            "search_threads": 1, "train_threads": _cpu_count(), "build_threads": _cpu_count(),
            "repeats": 3, "warmup": 1, "rss_interval": 0.005,
        },
        "cluster": {
            "coarse_cluster_count": 8 if smoke else 1000,
            "fine_cluster_count": 8 if smoke else 1000,
            "train_points_count": count, "kmeans_iterations_count": 2 if smoke else 10,
        },
        "puck": {"whether_pq": True, "nsq": 32, "filter_nsq": 8, "pq_train_points_count": count},
        "tinker": {
            "tinker_quantizer_type": "FLAT", "tinker_neighborhood_count": 16,
            "tinker_ef_construction": 600, "tinker_brief": False,
        },
        "search": {
            "hc": {
                "search_coarse_count": [4, 8] if smoke else [20, 50, 100, 200],
                "neighbors_count": [512] if smoke else [2000, 5000, 10000, 20000, 40000],
            },
            "puck": {
                "search_coarse_count": [4, 8] if smoke else [20, 50, 100, 200],
                "filter_topk": [200] if smoke else [200, 400, 800, 1600, 3200],
                "radius_rate": [1.0],
            },
            "tinker": {"ef_search": [100, 200] if smoke else [100, 200, 400, 800, 1600, 3200]},
        },
    }


def _merge(base, overrides, prefix=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate(config):
    for section in ("data", "gt", "runtime", "cluster", "puck", "tinker"):
        for key, value in config[section].items():
            if key in ("backend", "rss_interval", "seed", "whether_pq", "tinker_brief", "tinker_quantizer_type"):
                continue
            _positive_int(value, f"{section}.{key}")
    data = config["data"]
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("data.seed must be a non-negative integer")
    if data["count"] < data["k"] or data["count"] >= 2 ** 32 - 1:
        raise ValueError("require k <= count < uint32 sentinel")
    if config["gt"]["backend"] not in ("auto", "faiss", "numpy"):
        raise ValueError("gt.backend must be auto, faiss or numpy")
    interval = config["runtime"]["rss_interval"]
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 0 < interval <= 1:
        raise ValueError("rss_interval must be in (0, 1]")
    if config["runtime"]["repeats"] < 3:
        raise ValueError("at least three repeats are required")
    cluster = config["cluster"]
    if not max(cluster["coarse_cluster_count"], cluster["fine_cluster_count"]) <= cluster["train_points_count"]:
        raise ValueError("cluster counts exceed training points; use --smoke or a small-scale JSON config")
    if not cluster["train_points_count"] <= data["count"]:
        raise ValueError("train_points_count exceeds database size; use --smoke or adjust JSON config")
    puck = config["puck"]
    if puck["whether_pq"] is not True:
        raise ValueError("this benchmark requires Puck whether_pq=true")
    if not 256 <= puck["pq_train_points_count"] <= cluster["train_points_count"]:
        raise ValueError("PQ training needs 256 <= pq_train_points_count <= train_points_count")
    if any(data["dim"] % puck[key] != 0 for key in ("nsq", "filter_nsq")):
        raise ValueError("dim must be divisible by nsq and filter_nsq")
    if config["tinker"]["tinker_quantizer_type"] != "FLAT" or config["tinker"]["tinker_brief"] is not False:
        raise ValueError("this benchmark requires Tinker FLAT and brief=false")
    for engine, grid in config["search"].items():
        for key, values in grid.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"{engine}.{key} must be a non-empty list")
            for value in values:
                if key == "radius_rate":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value < math.inf:
                        raise ValueError("radius_rate must be finite and >= 1")
                else:
                    _positive_int(value, key)
                if key == "search_coarse_count" and value > cluster["coarse_cluster_count"]:
                    raise ValueError("search_coarse_count exceeds coarse clusters")
                if key in ("neighbors_count", "filter_topk", "ef_search") and value < data["k"]:
                    raise ValueError(f"{key} must be >= k")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory):
    return {
        str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": _file_hash(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _cached(directory, identity, compatible_identities=(), compatible_identity_predicate=None):
    """Validate a cache identity and every recorded file before reuse."""
    if not directory.exists():
        return False
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"incomplete cache: {directory}; use a new work directory")
    saved = _read_json(manifest)
    identities = (identity,) + tuple(compatible_identities)
    identity_matches = saved["identity"] in identities
    if not identity_matches and compatible_identity_predicate is not None:
        identity_matches = compatible_identity_predicate(saved["identity"])
    if not identity_matches:
        raise ValueError(f"cache mismatch: {directory}; no overwrite; use a new work directory")
    if saved["files"] != _inventory(directory):
        raise ValueError(f"cache contents changed: {directory}")
    result_file = directory / "result.json"
    if result_file.is_file():
        for name, digest in _read_json(result_file).get("native", {}).get("module_sha256", {}).items():
            path = Path(name)
            if not path.is_file() or _file_hash(path) != digest:
                raise ValueError(f"native/API module changed since measurement: {path}; use a new work directory")
    return True


def _begin(directory):
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir()
    return temporary


def _commit(temporary, directory, identity):
    _write_json(temporary / "manifest.json", {"identity": identity, "files": _inventory(temporary)})
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    temporary.rename(directory)


def _versions():
    versions = {}
    for name in ("numpy", "faiss-cpu", "matplotlib", "baidu-puck"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as distribution"
    return versions


def _hardware():
    cpu = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), "")
    return {
        "platform": platform.platform(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "available_cpus": _cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "hostname": platform.node(),
    }


def _metadata():
    revision = subprocess.run(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    ) if shutil.which("git") else None
    return {
        "python": sys.version, "hardware": _hardware(),
        "meminfo": Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else "",
        "versions": _versions(), "script_sha256": _file_hash(SCRIPT),
        "git_head": revision.stdout.strip() if revision is not None and revision.returncode == 0 else None,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _exact_topk(distances, ids, k):
    import numpy as np

    partition = np.argpartition(distances, k - 1, axis=1)[:, :k]
    best_dist = np.take_along_axis(distances, partition, axis=1)
    best_ids = np.take_along_axis(ids, partition, axis=1)
    # argpartition can choose arbitrary boundary ties. Resolve only that boundary,
    # avoiding a full sort of every query x database block.
    for row in range(len(distances)):
        threshold = best_dist[row].max()
        strict = best_dist[row] < threshold
        tied_ids = np.sort(ids[row, distances[row] == threshold])
        needed = k - int(strict.sum())
        best_ids[row, ~strict] = tied_ids[:needed]
    order = np.lexsort((best_ids, best_dist), axis=1)
    return np.take_along_axis(best_dist, order, axis=1), np.take_along_axis(best_ids, order, axis=1)


def _numpy_gt(database, queries, k, query_block, db_block):
    import numpy as np

    result = np.empty((len(queries), k), dtype=np.int64)
    # Float64 accumulation avoids norm=1 assumptions and reduces cancellation.
    for start in range(0, len(queries), query_block):
        query = queries[start:start + query_block].astype(np.float64)
        best_dist = np.full((len(query), k), np.inf)
        best_ids = np.full((len(query), k), -1, dtype=np.int64)
        for offset in range(0, len(database), db_block):
            block = database[offset:offset + db_block].astype(np.float64)
            distances = (query * query).sum(axis=1)[:, None] + (block * block).sum(axis=1)[None, :]
            distances -= 2 * query @ block.T
            ids = np.broadcast_to(np.arange(offset, offset + len(block)), distances.shape)
            distances = np.concatenate((best_dist, distances), axis=1)
            ids = np.concatenate((best_ids, ids), axis=1)
            best_dist, best_ids = _exact_topk(distances, ids, k)
        result[start:start + len(query)] = best_ids
    return result


def _prepare(config, output):
    import numpy as np

    data = config["data"]
    streams = np.random.SeedSequence(data["seed"]).spawn(2)
    vectors = []
    for count, seed in zip((data["count"], data["queries"]), streams):
        rng = np.random.default_rng(seed)
        array = rng.standard_normal((count, data["dim"]), dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("generated a zero vector; select another seed")
        array /= norms
        vectors.append(np.ascontiguousarray(array, dtype=np.float32))
    database, queries = vectors
    with (output / "db.fvecs").open("wb") as stream:
        for start in range(0, len(database), config["gt"]["db_block"]):
            block = database[start:start + config["gt"]["db_block"]]
            records = np.empty((len(block), data["dim"] + 1), dtype="<f4")
            records[:, 0].view("<i4")[:] = data["dim"]
            records[:, 1:] = block
            records.tofile(stream)
    np.save(output / "query.npy", queries, allow_pickle=False)
    backend = config["gt"]["backend"]
    faiss = None
    if backend in ("auto", "faiss"):
        try:
            import faiss
        except ImportError:
            if backend == "faiss":
                raise
    started = time.perf_counter()
    if faiss is not None:
        faiss.omp_set_num_threads(config["gt"]["threads"])
        index = faiss.IndexFlatL2(data["dim"])
        index.add(database)
        labels = np.empty((len(queries), data["k"]), dtype=np.int64)
        for start in range(0, len(queries), config["gt"]["query_block"]):
            block = queries[start:start + config["gt"]["query_block"]]
            _, labels[start:start + len(block)] = index.search(block, data["k"])
        actual_backend = "faiss.IndexFlatL2"
    else:
        labels = _numpy_gt(database, queries, data["k"], config["gt"]["query_block"], config["gt"]["db_block"])
        actual_backend = "numpy.float64.blocked_exact"
    np.save(output / "gt.npy", labels, allow_pickle=False)
    _write_json(output / "result.json", {
        "gt_backend": actual_backend, "gt_seconds": time.perf_counter() - started,
        "backend_version": getattr(faiss, "__version__", None) if faiss is not None else np.__version__,
        "metadata": _metadata(), "data": data,
        "distribution": "independent standard normal float32, L2-normalized once in prepare",
    })


def _engine_config(config, engine):
    flags = {"feature_dim": config["data"]["dim"], "whether_norm": False, "ip2cos": 0}
    if engine in ("hc", "puck"):
        flags.update(config["cluster"])
        flags["whether_pq"] = engine == "puck"
    if engine != "hc":
        flags.update(config[engine])
    if engine == "tinker":
        # Native defaults to 40 GiB and eagerly allocates capacity, even for smoke.
        width = 4 * (config["data"]["dim"] + config["tinker"]["tinker_neighborhood_count"] + 1)
        flags["tinker_memory_capacity"] = width * (config["data"]["count"] + 1)
    return flags


def _grid(config, engine):
    grid = config["search"][engine]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def _rss(pid):
    try:
        fields = Path(f"/proc/{pid}/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, ProcessLookupError):
        return 0


def _native_metadata(api):
    files = {}
    for name, module in tuple(sys.modules.items()):
        if "puck" in name and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if path.is_file():
                files[str(path)] = _file_hash(path)
    return {"module": api.__file__, "module_sha256": files}


def _search(index, config, search, output, data_dir):
    import numpy as np

    runtime = config["runtime"]
    queries = np.load(data_dir / "query.npy", allow_pickle=False)
    index.load()
    loaded_rss = _rss(os.getpid())
    arguments = {"top": config["data"]["k"], "ef_search": search.get("ef_search", 0)}
    for _ in range(runtime["warmup"]):
        distances, labels = index.search(queries, **arguments)
        del distances, labels
    repeats = []
    for repeat in range(runtime["repeats"]):
        before = _rss(os.getpid())
        _write_json(output / "window.json", {"repeat": repeat})
        started = time.perf_counter()
        distances, labels = index.search(queries, **arguments)
        seconds = time.perf_counter() - started
        after = _rss(os.getpid())
        (output / "window.json").unlink()
        np.save(output / f"labels-{repeat}.npy", labels, allow_pickle=False)
        repeats.append({
            "seconds": seconds, "qps": len(queries) / seconds,
            "rss_before_bytes": before, "rss_return_bytes": after,
            "return_arrays_bytes": distances.nbytes + labels.nbytes,
        })
        del distances, labels
    return {"loaded_rss_bytes": loaded_rss, "repeats": repeats}


def _worker(request):
    config = request["config"]
    output = Path(request["output"])
    stage = request["stage"]
    if stage == "prepare":
        _prepare(config, output)
        return
    if stage == "plot":
        _plot(Path(request["report"]), output)
        return
    import baidu_puck

    engine = request["engine"]
    index_dir = Path(request["index_dir"])
    flags = _engine_config(config, engine)
    flags.update({
        "index_path": str(index_dir), "feature_file_name": "db.fvecs",
        "train_fea_file_name": str(output / "scratch" / "train_clusters.dat"),
        "train_pq_file_name": str(output / "scratch" / "train_pq.dat"),
        "threads_count": config["runtime"][f"{stage}_threads"] if stage != "search" else 1,
        "context_initial_pool_size": config["runtime"]["search_threads"] if stage == "search" else 1,
        "topk": config["data"]["k"],
    })
    search = request.get("search", {})
    flags.update({key: value for key, value in search.items() if key != "ef_search"})
    (output / "scratch").mkdir(exist_ok=True)
    factory = "hierarchical_cluster" if engine == "hc" else engine
    index = getattr(baidu_puck, factory)(**flags)
    if stage == "search":
        result = _search(index, config, search, output, Path(request["data_dir"]))
    else:
        operation = getattr(index, stage)
        started = time.perf_counter()
        operation()
        result = {"seconds": time.perf_counter() - started}
    result.update({"metadata": _metadata(), "native": _native_metadata(baidu_puck),
                   "flags": flags, "config": config})
    _write_json(output / "result.json", result)


def _sample_window(output, pid, peaks, samples, rss_reader=None):
    """Count a PID's RSS sample only when the repeat marker brackets the read; return None."""
    if rss_reader is None:
        rss_reader = _rss
    try:
        repeat = _read_json(output / "window.json")["repeat"]
        rss = rss_reader(pid)
        confirmed_repeat = _read_json(output / "window.json")["repeat"]
    except FileNotFoundError:
        return
    if repeat == confirmed_repeat:
        peaks[repeat] = max(peaks.get(repeat, 0), rss)
        samples[repeat] = samples.get(repeat, 0) + 1


def _run_worker(request):
    output = Path(request["output"])
    _write_json(output / "request.json", request)
    environment = os.environ.copy()
    stage = request["stage"]
    runtime = request["config"]["runtime"]
    threads = runtime.get(f"{stage}_threads", request["config"]["gt"]["threads"])
    # Search parallelism belongs to the native wrapper, not BLAS/OpenMP.
    if stage == "search":
        threads = 1
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = str(threads)
    peaks = {}
    samples = {}
    with (output / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "_worker", str(output / "request.json")],
            cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                if stage == "search":
                    _sample_window(output, process.pid, peaks, samples)
                time.sleep(runtime["rss_interval"])
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode != 0:
        raise RuntimeError(f"{stage} worker exited {process.returncode}; see {output / 'worker.log'}")
    if stage == "search":
        result = _read_json(output / "result.json")
        for repeat, row in enumerate(result["repeats"]):
            row["sampled_rss_peak_bytes"] = max(
                peaks.get(repeat, 0), row["rss_before_bytes"], row["rss_return_bytes"])
            row["rss_samples"] = samples.get(repeat, 0)
        _write_json(output / "result.json", result)


def _recall(labels, truth, count, k):
    _positive_int(k, "k")
    if len(labels) != len(truth) or len(truth) == 0:
        raise ValueError("recall needs matching, non-empty query rows")
    hits = 0
    for returned, expected in zip(labels, truth):
        if len(returned) < k or len(expected) < k:
            raise ValueError("recall rows must have at least k labels")
        valid = {int(label) for label in returned[:k] if 0 <= int(label) < count}
        hits += len(valid.intersection(int(label) for label in expected[:k]))
    return hits / (len(truth) * k)


def _pareto(rows):
    ordered = sorted(rows, key=lambda row: (row["recall"], row["qps"]), reverse=True)
    frontier = []
    best_qps = -math.inf
    for row in ordered:
        if row["qps"] > best_qps:
            frontier.append(row)
            best_qps = row["qps"]
    return list(reversed(frontier))


def _identity(stage, config, dataset=None, engine=None, training_artifact=None):
    identity = {"schema": SCHEMA, "stage": stage}
    if stage != "prepare":
        identity["script"] = _file_hash(SCRIPT)
    if stage == "prepare":
        versions = _versions()
        identity.update({"data": config["data"], "gt": config["gt"],
                         "data_semantics": {
                             "normalization": "l2_once_float32",
                             "metric": "l2",
                         },
                         "versions": {name: versions[name] for name in ("numpy", "faiss-cpu")}})
    else:
        identity.update({
            "dataset": dataset, "engine": engine, "flags": _engine_config(config, engine),
            "train_threads": config["runtime"]["train_threads"],
            "implementation": _implementation_hash(),
            "python": sys.version, "versions": _versions(), "hardware": _hardware(),
        })
        if stage == "build":
            identity["build_threads"] = config["runtime"]["build_threads"]
            identity["training_artifact"] = training_artifact
    return identity


def _legacy_prepare_identity(config):
    """Return the pre-change prepare identity for valid old data manifests."""
    versions = _versions()
    return {
        "schema": SCHEMA,
        "stage": "prepare",
        "script": _file_hash(SCRIPT),
        "data": config["data"],
        "gt": config["gt"],
        "versions": {name: versions[name] for name in ("numpy", "faiss-cpu")},
    }


def _legacy_prepare_identity_matches(saved, config):
    """Match old prepare identities while ignoring only their obsolete script hash."""
    if not isinstance(saved, dict):
        return False
    expected = _legacy_prepare_identity(config)
    saved_without_script = dict(saved)
    saved_without_script.pop("script", None)
    expected.pop("script", None)
    return saved_without_script == expected


def _implementation_hash():
    """Return source/native fingerprint without importing the extension."""
    files = {}
    for folder in ("baidu_puck", "pyapi_wrapper", "interface", "gflags", "hierarchical_cluster",
                   "puck", "tinker", "quantizer", "common"):
        for path in sorted((PROJECT / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".so", ".i"):
                files[str(path.relative_to(PROJECT))] = _file_hash(path)
    for path in sorted(PROJECT.glob("*.so")):
        files[path.name] = _file_hash(path)
    return _digest(files)


def _data_ready(work, config):
    directory = work / "data"
    if not _cached(directory, _identity("prepare", config),
                   compatible_identity_predicate=lambda saved: _legacy_prepare_identity_matches(
                       saved, config)):
        raise ValueError("prepare must complete first")
    return _digest(_read_json(directory / "manifest.json"))


def _build_identity(work, config, dataset, engine):
    """Validate training cache and return build identity linked to its manifest; raise on mismatch."""
    trained = work / engine / "train"
    if not _cached(trained, _identity("train", config, dataset, engine)):
        raise ValueError(f"train {engine} first")
    artifact = _digest(_read_json(trained / "manifest.json"))
    return _identity("build", config, dataset, engine, training_artifact=artifact)


def _lifecycle(work, config, engines, stage):
    dataset = _data_ready(work, config)
    for engine in engines:
        identity = _identity(stage, config, dataset, engine)
        if stage == "build":
            identity = _build_identity(work, config, dataset, engine)
        destination = work / engine / stage
        if _cached(destination, identity):
            print(f"reuse {destination}")
            continue
        trained = work / engine / "train"
        output = _begin(destination)
        index_dir = output / "index"
        if stage == "build":
            # Copy codebooks, never hard-link mutable training/build artifacts.
            shutil.copytree(trained / "index", index_dir, symlinks=True)
        else:
            index_dir.mkdir()
            (index_dir / "db.fvecs").symlink_to(work / "data" / "db.fvecs")
        _run_worker({
            "stage": stage, "config": config, "engine": engine, "output": str(output),
            "index_dir": str(index_dir),
        })
        result = _read_json(output / "result.json")
        _write_csv(output / "timing.csv", [{"engine": engine, "stage": stage, "seconds": result["seconds"]}],
                   ("engine", "stage", "seconds"))
        _commit(output, destination, identity)


def _search_all(work, config, engines):
    import numpy as np

    dataset = _data_ready(work, config)
    truth = np.load(work / "data" / "gt.npy", allow_pickle=False)
    rows = []
    stages = []
    for engine in engines:
        built = work / engine / "build"
        if not _cached(built, _build_identity(work, config, dataset, engine)):
            raise ValueError(f"build {engine} first")
        artifact = _digest(_read_json(built / "manifest.json"))
        for stage in ("train", "build"):
            result = _read_json(work / engine / stage / "result.json")
            stages.append({"engine": engine, "stage": stage, "seconds": result["seconds"],
                           "metadata": result["metadata"], "native": result["native"]})
        for search in _grid(config, engine):
            identity = {
                "schema": SCHEMA, "script": _file_hash(SCRIPT), "artifact": artifact,
                "search": search, "runtime": config["runtime"],
                "hardware": _hardware(), "python": sys.version, "versions": _versions(),
            }
            destination = work / engine / "search" / _digest(identity)[:20]
            if not _cached(destination, identity):
                output = _begin(destination)
                _run_worker({
                    "stage": "search", "config": config, "engine": engine, "output": str(output),
                    "index_dir": str(built / "index"), "data_dir": str(work / "data"), "search": search,
                })
                result = _read_json(output / "result.json")
                for repeat, row in enumerate(result["repeats"]):
                    labels = np.load(output / f"labels-{repeat}.npy", allow_pickle=False)
                    row["recall"] = _recall(labels, truth, config["data"]["count"], config["data"]["k"])
                result.update({
                    "engine": engine, "search": search,
                    "recall": statistics.median(row["recall"] for row in result["repeats"]),
                    "qps": statistics.median(row["qps"] for row in result["repeats"]),
                    "search_rss_bytes": max(row["sampled_rss_peak_bytes"] for row in result["repeats"]),
                })
                _write_json(output / "result.json", result)
                _write_csv(output / "repeats.csv", result["repeats"], tuple(result["repeats"][0]))
                _commit(output, destination, identity)
            rows.append(_read_json(destination / "result.json"))
    report_identity = {"config": config, "engines": engines, "dataset": dataset, "rows": _digest(rows)}
    report = work / "reports" / _digest({"config": config, "engines": engines})[:20]
    if not _cached(report, report_identity):
        output = _begin(report)
        _write_json(output / "summary.json", {
            "config": config, "engines": engines, "dataset": dataset, "search": rows, "stages": stages,
            "metadata": _metadata(),
            "rss_definition": "sampled search-window RSS, not exact peak; includes return arrays",
        })
        _write_csv(output / "search.csv", rows,
                   ("engine", "search", "recall", "qps", "loaded_rss_bytes", "search_rss_bytes"))
        _write_csv(output / "stages.csv", stages, ("engine", "stage", "seconds"))
        repeat_rows = [
            dict(row, engine=result["engine"], search=result["search"], repeat=repeat)
            for result in rows for repeat, row in enumerate(result["repeats"])
        ]
        _write_csv(output / "repeats.csv", repeat_rows, tuple(repeat_rows[0]))
        _commit(output, report, report_identity)
    print(f"report: {report}")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                             for key, value in row.items()})


def _plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = _read_json(report / "summary.json")
    for metric, ylabel, name in (
        ("qps", "Queries / second (median)", "recall-qps"),
        ("search_rss_bytes", "Search-window sampled RSS (MiB)", "recall-rss"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for engine in summary["engines"]:
            rows = [row for row in summary["search"] if row["engine"] == engine]
            divisor = 1024 ** 2 if metric == "search_rss_bytes" else 1
            points = axis.scatter([row["recall"] for row in rows],
                                  [row[metric] / divisor for row in rows], label=engine, alpha=0.7)
            if metric == "qps":
                frontier = _pareto(rows)
                axis.plot([row["recall"] for row in frontier], [row["qps"] for row in frontier],
                          color=points.get_facecolor()[0], linewidth=1)
        if metric == "qps":
            axis.set_yscale("log")
        axis.set(xlabel=f"Recall@{summary['config']['data']['k']}", ylabel=ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(summary["engines"]))
    for offset, stage in enumerate(("train", "build")):
        seconds = [next(row["seconds"] for row in summary["stages"]
                        if row["engine"] == engine and row["stage"] == stage) for engine in summary["engines"]]
        axis.bar(positions + (offset - 0.5) * 0.36, seconds, 0.36, label=stage)
    axis.set_xticks(positions)
    axis.set_xticklabels(summary["engines"])
    axis.set_ylabel("API wall time (seconds)")
    axis.legend()
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(output / f"train-build.{extension}", dpi=160)
    plt.close(figure)


def _execute(args, config):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # Serialize writers and protect directory publication from concurrent commands.
    import fcntl

    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            identity = _identity("prepare", config)
            destination = work / "data"
            if _cached(destination, identity,
                       compatible_identity_predicate=lambda saved: _legacy_prepare_identity_matches(
                           saved, config)):
                print(f"reuse {destination}")
                return
            print(f"Exact GT: {config['data']['count'] * config['data']['queries']:,} distances; "
                  "full defaults are expensive (10 billion pairs).", flush=True)
            output = _begin(destination)
            _run_worker({"stage": "prepare", "config": config, "output": str(output)})
            _commit(output, destination, identity)
        elif args.stage in ("train", "build"):
            _lifecycle(work, config, args.engines, args.stage)
        elif args.stage == "search":
            _search_all(work, config, args.engines)
        else:
            report = work / "reports" / _digest({"config": config, "engines": args.engines})[:20]
            if not (report / "manifest.json").exists():
                raise ValueError("matching search report required before plot")
            saved = _read_json(report / "manifest.json")
            _cached(report, saved["identity"])
            destination = work / "plots" / report.name
            identity = {"report": _digest(saved), "script": _file_hash(SCRIPT)}
            if not _cached(destination, identity):
                output = _begin(destination)
                _run_worker({"stage": "plot", "config": config, "output": str(output), "report": str(report)})
                _commit(output, destination, identity)
            print(f"plots: {destination}")


def main():
    """Run sys.argv's stage; return None, raising ValueError/RuntimeError on invalid input or worker failure."""
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        _worker(_read_json(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "build", "search", "plot"))
    parser.add_argument("--work", type=Path, required=True, help="exclusive benchmark artifact directory")
    parser.add_argument("--config", type=Path, help="strict nested JSON overrides")
    parser.add_argument("--smoke", action="store_true",
                        help="4096 db / 32 queries / 8x8 clusters; not a full benchmark")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--search-threads", type=int)
    parser.add_argument("--train-threads", type=int)
    parser.add_argument("--build-threads", type=int)
    args = parser.parse_args()
    config = _defaults(args.smoke)
    if args.config is not None:
        _merge(config, _read_json(args.config))
    for key in ("search_threads", "train_threads", "build_threads"):
        if getattr(args, key) is not None:
            config["runtime"][key] = getattr(args, key)
    _validate(config)
    if len(set(args.engines)) != len(args.engines):
        raise ValueError("duplicate engines")
    _execute(args, config)


if __name__ == "__main__":
    main()
