#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copyright (c) 2026 Baidu, Inc. All Rights Reserved.

Reproducible, subprocess-isolated benchmark for the real baidu_puck API.
Maintainer: Puck contributors (contact: repository issue tracker).
See benchmark.md for measurement definitions and operational guidance.
"""

import argparse
import ctypes
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
                "radius_rate": [1.05],
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


def _validate_artifact(directory, required_files=(), required_directories=()):
    """Validate an upstream artifact without comparing execution identities."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing upstream artifact: {directory}")
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"incomplete upstream artifact: {directory}")
    saved = _read_json(manifest)
    if saved.get("files") != _inventory(directory):
        raise ValueError(f"upstream artifact contents changed: {directory}")
    for name in required_files:
        if not (directory / name).is_file():
            raise ValueError(f"missing upstream file: {directory / name}")
    for name in required_directories:
        if not (directory / name).is_dir():
            raise ValueError(f"missing upstream directory: {directory / name}")
    return saved


def _begin(directory):
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir()
    return temporary


def _commit(temporary, directory):
    """Publish using Linux RENAME_EXCHANGE; keep old artifacts without deleting data."""
    _write_json(temporary / "manifest.json", {"schema": SCHEMA, "files": _inventory(temporary)})
    if not directory.exists():
        os.replace(temporary, directory)
        return
    # Two renames via a backup leave a visibility/crash gap. Exchange has none.
    libc = ctypes.CDLL(None, use_errno=True)
    rename_exchange = libc.renameat2
    rename_exchange.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename_exchange.restype = ctypes.c_int
    if rename_exchange(-100, os.fsencode(temporary), -100, os.fsencode(directory), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(directory))
    # The old generation now occupies temporary. Retain it, including data files.
    # Publication already succeeded; housekeeping must not turn success into failure.
    try:
        temporary.rename(directory.with_name(directory.name + ".old-" + uuid.uuid4().hex))
    except OSError:
        pass


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
        "versions": _versions(),
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
    return {"module": api.__file__}


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


def _upstream_data(work, artifact, config):
    """Resolve new snapshots or legacy shared data, checking integrity and dimensions."""
    directory = artifact / "data" if (artifact / "data").exists() else work / "data"
    _validate_artifact(directory, ("db.fvecs", "query.npy", "gt.npy", "result.json"))
    actual = _read_json(directory / "result.json").get("data", {})
    for key in ("count", "queries", "dim", "k"):
        if actual.get(key) != config["data"][key]:
            raise ValueError(
                f"data size mismatch: {key}: CLI={config['data'][key]}, "
                f"data={actual.get(key)} ({directory / 'result.json'})")
    return directory


def _data_metadata(directory):
    """Read prepare metadata and validate query dimensions before search."""
    import numpy as np

    result = _read_json(directory / "result.json")
    data = result.get("data", {})
    queries = np.load(directory / "query.npy", allow_pickle=False)
    if queries.ndim != 2 or queries.shape[1] != data["dim"] or queries.shape[0] != data["queries"]:
        raise ValueError(f"query dimensions do not match data metadata: {directory / 'query.npy'}")
    if data["count"] <= 0 or data["k"] > data["count"]:
        raise ValueError(f"invalid data count/k metadata: {directory / 'result.json'}")
    return data


def _training_result(work, engine, built):
    """Read optional legacy timing without rewriting the upstream manifest."""
    path = built / "train-result.json"
    if not path.is_file():
        trained = work / engine / "train"
        path = trained / "result.json"
        if not path.is_file():
            return {}
        _validate_artifact(trained, ("result.json",))
    return _read_json(path)


def _lifecycle(work, config, engines, stage):
    if stage == "train":
        _validate_artifact(work / "data", ("db.fvecs", "query.npy", "gt.npy", "result.json"))
    else:
        for engine in engines:
            _validate_artifact(work / engine / "train", ("result.json", "index/db.fvecs"), ("index",))
    sources = {
        engine: _upstream_data(work, work / engine / "train" if stage == "build" else work, config)
        for engine in engines
    }
    for engine in engines:
        destination = work / engine / stage
        trained = work / engine / "train"
        output = _begin(destination)
        index_dir = output / "index"
        if stage == "build":
            # Copy mutable native artifacts; relative data link stays inside the snapshot.
            shutil.copytree(trained / "index", index_dir, symlinks=True,
                            ignore=shutil.ignore_patterns("db.fvecs"))
            shutil.copytree(sources[engine], output / "data")
            (index_dir / "db.fvecs").symlink_to("../data/db.fvecs")
            shutil.copyfile(trained / "result.json", output / "train-result.json")
        else:
            # Self-contained snapshots let downstream stages read only their predecessor.
            shutil.copytree(sources[engine], output / "data")
            index_dir.mkdir()
            (index_dir / "db.fvecs").symlink_to("../data/db.fvecs")
        _run_worker({
            "stage": stage, "config": config, "engine": engine, "output": str(output),
            "index_dir": str(index_dir),
        })
        result = _read_json(output / "result.json")
        _write_csv(output / "timing.csv", [{"engine": engine, "stage": stage, "seconds": result["seconds"]}],
                   ("engine", "stage", "seconds"))
        _commit(output, destination)


def _search_all(work, config, engines):
    import numpy as np

    for engine in engines:
        _validate_artifact(work / engine / "build", ("result.json", "index/db.fvecs"), ("index",))
    sources = {engine: _upstream_data(work, work / engine / "build", config) for engine in engines}
    report = work / "reports" / _digest({"config": config, "engines": engines})[:20]
    output = _begin(report)
    rows = []
    stages = []
    dataset = {}
    for engine in engines:
        built = work / engine / "build"
        data_dir = sources[engine]
        data_info = _data_metadata(data_dir)
        truth = np.load(data_dir / "gt.npy", allow_pickle=False)
        if truth.ndim != 2 or truth.shape != (data_info["queries"], data_info["k"]):
            raise ValueError(f"ground-truth dimensions do not match data metadata: {data_dir / 'gt.npy'}")
        dataset[engine] = data_info
        for stage in ("train", "build"):
            result = (_training_result(work, engine, built) if stage == "train"
                      else _read_json(built / "result.json"))
            stages.append({"engine": engine, "stage": stage, "seconds": result.get("seconds"),
                           "metadata": result.get("metadata", {}), "native": result.get("native", {})})
        for point_number, search in enumerate(_grid(config, engine)):
            point = output / "points" / engine / str(point_number)
            point.mkdir(parents=True)
            _run_worker({
                "stage": "search", "config": config, "engine": engine, "output": str(point),
                "index_dir": str(built / "index"), "data_dir": str(data_dir), "search": search,
            })
            result = _read_json(point / "result.json")
            for repeat, row in enumerate(result["repeats"]):
                labels = np.load(point / f"labels-{repeat}.npy", allow_pickle=False)
                row["recall"] = _recall(labels, truth, data_info["count"], data_info["k"])
            result.update({
                "engine": engine, "search": search,
                "recall": statistics.median(row["recall"] for row in result["repeats"]),
                "qps": statistics.median(row["qps"] for row in result["repeats"]),
                "search_rss_bytes": max(row["sampled_rss_peak_bytes"] for row in result["repeats"]),
            })
            _write_json(point / "result.json", result)
            _write_csv(point / "repeats.csv", result["repeats"], tuple(result["repeats"][0]))
            rows.append(result)
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
    _commit(output, report)
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
        figure, axis = plt.subplots(figsize=(12, 7))
        for engine in summary["engines"]:
            rows = [row for row in summary["search"] if row["engine"] == engine]
            divisor = 1024 ** 2 if metric == "search_rss_bytes" else 1
            color = f"C{summary['engines'].index(engine)}"
            frontier = sorted(_pareto(rows), key=lambda row: row["recall"])
            axis.scatter(
                [row["recall"] for row in frontier],
                [row[metric] / divisor for row in frontier],
                color=color,
                label=engine,
                alpha=0.7,
            )
            for point_index, row in enumerate(frontier):
                search = row["search"]
                if engine == "hc":
                    text = (f"c{search['search_coarse_count']},"
                            f"n{search['neighbors_count'] // 1000}k")
                elif engine == "puck":
                    text = (f"c{search['search_coarse_count']},"
                            f"f{search['filter_topk']}")
                    if search["radius_rate"] != 1.0:
                        text += f",r{search['radius_rate']:g}"
                else:
                    text = f"ef{search['ef_search']}"
                offset_pattern = (
                    (-10, 18), (10, 18), (-10, -24), (10, -24),
                    (-18, 34), (18, 34), (-18, -40), (18, -40),
                )
                offset_index = (point_index + summary["engines"].index(engine) * 2) % len(offset_pattern)
                offset_x, offset_y = offset_pattern[offset_index]
                axis.annotate(
                    text, (row["recall"], row[metric] / divisor),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    ha="right" if offset_x < 0 else "left",
                    va="bottom" if offset_y > 0 else "top",
                    fontsize=7, color=color,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5},
                )
            if len(frontier) > 1:
                axis.plot(
                    [row["recall"] for row in frontier],
                    [row[metric] / divisor for row in frontier],
                    color=color,
                    linewidth=2.5,
                    alpha=0.95,
                )
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
        seconds = [next((row.get("seconds") for row in summary["stages"]
                         if row["engine"] == engine and row["stage"] == stage), None)
                   for engine in summary["engines"]]
        locations = positions + (offset - 0.5) * 0.36
        axis.bar(locations, [value if value is not None else float("nan") for value in seconds],
                 0.36, label=stage)
        for location, value in zip(locations, seconds):
            if value is None:
                axis.annotate("N/A", (location, 0), xytext=(0, 4), textcoords="offset points", ha="center")
    axis.set_xticks(positions)
    axis.set_xticklabels(summary["engines"])
    axis.set_ylabel("API wall time (seconds)")
    axis.legend()
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(output / f"train-build.{extension}", dpi=160)
    plt.close(figure)


def _find_report(work, config, engines):
    """Locate the report by configuration name, validating only file integrity."""
    report = work / "reports" / _digest({"config": config, "engines": engines})[:20]
    _validate_artifact(report, ("summary.json", "search.csv", "stages.csv", "repeats.csv"))
    return report


def _execute(args, config):
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    # Serialize writers and protect directory publication from concurrent commands.
    import fcntl

    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            destination = work / "data"
            print(f"Exact GT: {config['data']['count'] * config['data']['queries']:,} distances; "
                  "full defaults are expensive (10 billion pairs).", flush=True)
            output = _begin(destination)
            _run_worker({"stage": "prepare", "config": config, "output": str(output)})
            _commit(output, destination)
        elif args.stage in ("train", "build"):
            _lifecycle(work, config, args.engines, args.stage)
        elif args.stage == "search":
            _search_all(work, config, args.engines)
        else:
            report = getattr(args, "report", None)
            if report is None:
                report = _find_report(work, config, args.engines)
            else:
                report = report.resolve()
                _validate_artifact(report, ("summary.json",))
            destination = work / "plots" / report.name
            output = _begin(destination)
            _run_worker({"stage": "plot", "config": config, "output": str(output), "report": str(report)})
            _commit(output, destination)
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
    parser.add_argument("--report", type=Path, help="plot an explicit report directory, ignoring config lookup")
    parser.add_argument("--smoke", action="store_true",
                        help="4096 db / 32 queries / 8x8 clusters; not a full benchmark")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--search-threads", type=int)
    parser.add_argument("--train-threads", type=int)
    parser.add_argument("--build-threads", type=int)
    args = parser.parse_args()
    if args.report is not None and args.stage != "plot":
        parser.error("--report is only supported by plot")
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
