# Puck Python API benchmark

本脚本测量真实 `baidu_puck.hierarchical_cluster/puck/tinker` 的
`train()/build()/load()/search()`，不替换 native、不修改算法。

## 环境要求

* Linux（依赖 `/proc` 与 `fcntl`）、Python 3.9+、NumPy。
* 与当前 Python ABI 匹配、可导入的真实 baidu_puck SWIG/native 扩展。
* FAISS CPU 可选：`auto` 优先用 `faiss.IndexFlatL2`，导入不可用时回退到精确 NumPy；
  显式指定 `faiss` 不可用则报错。`plot` 阶段需要 Matplotlib。
* 依赖延迟导入，`--help` 不需要这些依赖；脚本不安装或编译任何依赖。

## 运行步骤

各阶段按 `prepare → train → build → search → plot` 顺序执行，每个阶段是独立子进程。
以下变量指向绝对路径，在同一个 shell 中依次执行：

```bash
BENCH=/path/to/puck/test/benchmark.py
WORK=/path/to/puck/benchmark-smoke-v2
python3 "$BENCH" --help
python3 "$BENCH" prepare --work "$WORK" --smoke --train-threads 2 --build-threads 2
python3 "$BENCH" train   --work "$WORK" --smoke --train-threads 2 --build-threads 2
# 仅 build_threads 变化：此命令复用已完成训练缓存
python3 "$BENCH" train   --work "$WORK" --smoke --train-threads 2 --build-threads 4
python3 "$BENCH" build   --work "$WORK" --smoke --train-threads 2 --build-threads 4
python3 "$BENCH" search  --work "$WORK" --smoke --train-threads 2 --build-threads 4
python3 "$BENCH" plot    --work "$WORK" --smoke --train-threads 2 --build-threads 4
```

* **smoke**：4096 db / 32 queries / 8×8 簇，仅用于验证流程，不用于性能评估。
* **全量**：使用新的 WORK 绝对路径，并从每条命令去掉 `--smoke`；不要直接
  把 smoke 目录用于全量。默认 db 1,000,000×32、query 10,000×32、recall@100，
  GT 约 100 亿向量对，全量运行前须评估 CPU/内存/磁盘预算。
* `--engines hc puck` 可选引擎子集，默认运行全部三个引擎。
* `--search-threads` 默认 1，直接设置 native `context_initial_pool_size`，
  不是 Python 线程池；train/build 线程可分别设置，默认取进程 affinity 内可用 CPU 数。

## 配置覆盖（可选）

`--config /absolute/path/config.json` 严格嵌套覆盖默认配置，未知 key 报错；
合并顺序：默认值 → `--smoke` 默认值 → JSON → 命令行线程参数。示例：

```json
{
  "data": {"count": 4096, "queries": 32, "dim": 32, "seed": 1234, "k": 100},
  "gt": {"backend": "numpy", "query_block": 16, "db_block": 1024, "threads": 2},
  "cluster": {
    "coarse_cluster_count": 8, "fine_cluster_count": 8,
    "train_points_count": 4096, "kmeans_iterations_count": 2
  },
  "puck": {"nsq": 32, "filter_nsq": 8, "pq_train_points_count": 4096},
  "runtime": {"train_threads": 2, "build_threads": 2, "search_threads": 1},
  "search": {
    "hc": {"search_coarse_count": [4, 8], "neighbors_count": [512]},
    "puck": {"search_coarse_count": [4, 8], "filter_topk": [200]},
    "tinker": {"ef_search": [100, 200]}
  }
}
```

脚本运行前会校验配置，例如：簇数 ≤ 训练点数 ≤ db 数、PQ 训练点 ≥ 256、
dim 可被 nsq/filter_nsq 整除、search_coarse_count ≤ 粗簇数、k ≤ count。
维度变化时需同步调整 nsq/filter_nsq。

## 缓存与重跑

* 每个成功阶段按配置/线程/实现指纹缓存；重跑相同配置直接复用，不重新计算。
* **强制重跑须使用新的 WORK**；脚本不提供 `--force`，不覆盖已有产物。
* train 完成后再调整 `build_threads` 可复用训练缓存；改变 `train_threads`
  会使 train/build 同时失效。search 线程数与搜索网格不影响 train/build。
* 修改 benchmark 脚本本身不影响已生成的 data（prepare 缓存不依赖脚本指纹），
  但 train/build/search 仍严格校验脚本/实现指纹，不匹配会报错而非静默重算。
* plot 的配置必须与对应 search 一致；某阶段失败时查看该阶段的 `worker.log`。

## 输出

* 各阶段产物与 manifest 写入 WORK 下对应目录；report 与 plots 使用配置摘要
  子目录，命令会打印绝对路径。
* search 产出 CSV（search/stages/repeats 三表）与 JSON（配置、硬件、版本、
  API flags、各 repeat 时间/QPS/recall/RSS）；plot 生成 PNG/SVG
  （recall-QPS 含各引擎 Pareto 前沿、recall-RSS、train/build 柱状图）。
