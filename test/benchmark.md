# Puck Python API benchmark

本脚本测量 `baidu_puck.hierarchical_cluster/puck/tinker` 的
`train()/build()/load()/search()` Python API 性能。
默认使用 100 万条 32 维 db 向量和 1 万条独立 query 向量，统一 L2 归一化，
以 L2 距离计算 recall@100，并比较 QPS、训练/建库耗时和搜索内存占用。

## 环境要求

见dockerfile

## 运行步骤


先设置脚本和工作目录：

```bash
BENCH=/path/to/puck/test/benchmark.py
WORK=/path/to/puck/benchmark
python3 "$BENCH" --help
```

### 1. prepare：准备数据和 ground truth

```bash
python3 "$BENCH" prepare --work "$WORK"
```

该阶段按配置生成随机 db/query 向量，统一做 L2 归一化，生成特征文件、query 文件和精确 Top-K ground truth，并将结果写入 `WORK/data/`，供后续训练阶段读取。

### 2. train：码本训练

```bash
python3 "$BENCH" train --work "$WORK"
```

该阶段分别训练三种索引所需的模型数据：HC 训练层次聚类码本，
Puck 在此基础上训练 PQ 码本，Tinker 准备并保存 FLAT 量化器配置，
同时记录训练耗时，结果写入 `WORK/<engine>/train/`。

### 3. build：构建索引

```bash
python3 "$BENCH" build --work "$WORK"
```

该阶段使用上一步训练得到的模型和数据库向量，为 HC、Puck、Tinker 分别建立可检索索引，并记录建库耗时。索引建立完成后，结果保存在 `WORK/<engine>/build/`，供下一阶段搜索使用。

### 4. search：执行搜索 benchmark

```bash
python3 "$BENCH" search --work "$WORK" --search-threads 1
```

该阶段加载各引擎的 build 产物，按照配置扫描 HC、Puck、Tinker 的搜索参数组合，执行预热和多次正式搜索，计算每个参数点的 QPS、recall@K 和搜索 RSS，并生成搜索明细、重复测量结果和汇总 report。

### 5. plot：绘制结果图

```bash
python3 "$BENCH" plot --work "$WORK"
```

该阶段读取 search 产生的 report，绘制 recall-QPS、recall-RSS 和 train/build 耗时图，在 recall-QPS 图中突出各引擎的 Pareto 前沿并标注搜索参数，最后将 PNG/SVG 图片写入 `WORK/plots/`。

如只需重绘已有 report，可直接指定 report 路径：

```bash
python3 "$BENCH" plot --work "$WORK" --report /absolute/path/to/report
```

以下参数适用于上述各阶段：

* `prepare` 默认需要为 1 万条查询向量分别计算与 100 万条数据库向量的距离，
  共约 100 亿次距离计算，以获得精确近邻结果。该步骤耗时较长，请预留计算资源和磁盘空间；
  数据准备完成后，后续训练、建库和搜索无需重新执行 `prepare`。
* `--engines hc puck` 可选引擎子集，默认运行全部三个引擎。
* `--search-threads` 默认 1；train/build 线程可分别设置，默认取进程 affinity 内可用 CPU 数。

## 配置覆盖（可选）

`--config /absolute/path/config.json` 严格嵌套覆盖默认配置，未知 key 报错；
未指定的字段保留默认值，命令行线程参数优先于 JSON。以下示例保留百万级数据规模，
设置训练/建库线程数，并自定义搜索参数扫描范围：

```json
{
  "runtime": {"train_threads": 16, "build_threads": 16, "search_threads": 1},
  "search": {
    "hc": {"search_coarse_count": [20, 50, 100, 200], "neighbors_count": [2000, 5000, 10000, 20000, 40000]},
    "puck": {"search_coarse_count": [20, 50, 100, 200], "filter_topk": [200, 400, 800, 1600, 3200], "radius_rate": [1.05]},
    "tinker": {"ef_search": [100, 200, 400, 800, 1600, 3200]}
  }
}
```
