# Changelog

本文件记录本仓库所有重要变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased] — 2026-09-21

> Tinker 图索引全面重构，三种索引统一接口，仓库模块化重组并升级构建标准，补齐 Python 包与测试。

### Added

- **统一索引接口**：Puck、HierarchicalCluster、Tinker 三种索引对外暴露同一套训练、建库、加载、检索流程，切换引擎只需换一个名字
- **Tinker 动态量化**：支持自配置量化器（FLAT、LVQ）
- **Tinker 增量插入与标签过滤**：可在线向图索引插入(带标签)数据；检索时可按标签过滤结果
- **Python 包 `baidu_puck`**：三种索引均可直接通过 Python 使用：`train`/`build`/`load`/`search`，完善Python API 文档
- **测试**：新增Tinker索引相关的测试用例

### Changed

- **强化topK gflag配置项语义**：`puck` 和 `HierarchicalCluster`以topk gflag配置项为最大召回数量限制，`tinker` 没有这个限制
- **Python 接口重做**：构造时指定索引类型；原"一键初始化"拆分为训练、建库、加载三个独立步骤；搜索接口增加宽度参数，并加强入参校验，参数非法时明确报错而非静默失败
- **构建体系升级**：编译标准升级到 C++20；源码按模块重组，量化器与图索引拆为独立组件，工具链、示例、Python 绑定统一链接主库
- **安装包更名**：Python 安装包由 `puck` 更名为 `baidu-puck`

### Fixed

- **索引落盘失败不再静默**：此前模型文件写入失败（如磁盘空间不足）可能静默留下损坏索引，现在会明确报错并终止流程
- **索引类型识别更轻量可靠**：直接从索引文件头部读取类型标识，不再整份加载模型来判断，速度更快、占用更小
- **强化insert场景单元测试**：并发插入改为分批执行并逐条校验结果，能捕获漏插与失败；外部命令执行也校验退出状态

### Removed

- **旧 Tinker 索引实现**：原基于hnswlib的实现整体删除，替换为自研实现
