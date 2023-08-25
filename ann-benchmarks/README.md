## ANN-Benchmarks
该benchmark是fork自https://github.com/NJU-yasuo/big-ann-benchmarks。

在其基础上，修改如下

1.检索内存限制60G

2.建库内存现在160G

3.支持faiss的index_factory建库（T1）

4.支持Nmslib（hnsw在业内最好的实现版本）

5.支持Puck&Tinker

6.增加多种算法在bigann-10M、deep-10M、bigann-100M、deep-100M四个数据集上的训练建库和检索配置（faiss::IVF-Flat、faiss::IVF-PQ、faiss::HNSW、Nmslib、Puck、Puck-Flat、Tinker）

### 使用方法

安装、使用和对比方法与https://github.com/NJU-yasuo/big-ann-benchmarks 保持一致。

### benchmark
以下对标数据的考量标准如下：

1.QPS=1W时, 对应的Top100召回率，越高越好，

2.召回率在[0.8, 0.95)范围内，相同召回率下的QPS，越高越好

#### BIGANN-100M



|   | Faiss-IVF  | Faiss-IVFPQ  |  Faiss-HNSW | Nmslib(HNSW)  | Puck  | Puck-Flat  | Tinker  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Build Params  | IVF100000,Flat  | IVF100000,PQ96  | HNSW32,Flat, "efConstruction" : 640  | "M": 32, "efConstruction": 640, "indexThreadQty":32  | "C":3000, "F":3000, "FN":16, "N":128  |  "C":3000, "F":3000, "FN":16, "N":0 | "C":3000, "F":3000,"tinker_neighborhood":10,"tinker_consstruction":600  |
| Index Size(G)  | 48.5399  | 12.726  |   |   | 14.6548  | 50.0447  | 56.077  |
| Recall （QPS=1W）  | 0.634  |  0.405214 | 60G检索内存超限  | 160G建库内存超限  | 0.881690  | 0.868196  | 0.926411  |
| QPS (Recall = 0.90)  | 1802  | 1156  |   |   | 9234  | 9347  | 13590  |

QPS & 召回率

![bigann-100M](../ann-benchmarks/results/bigann-100M.png)


#### DEEP-100M



|   | Faiss-IVF  | Faiss-IVFPQ  |  Faiss-HNSW | Nmslib(HNSW)  | Puck  | Puck-Flat  | Tinker  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Build Params  | IVF100000,Flat  | IVF100000,PQ96  | HNSW32,Flat, "efConstruction" : 640  | "M": 32, "efConstruction": 640, "indexThreadQty":32  | "C":3000, "F":3000, "FN":16, "N":96  |  "C":3000, "F":3000, "FN":16, "N":0 | "C":3000, "F":3000,"tinker_neighborhood":16,"tinker_construction":600  |
| Index Size(G)  | 36.5807  | 9.72923  |   |   | 11.6738  | 38.1231  | 48.625  |
| Recall （QPS=1W）  | 0.724  |  0.675 | 60G检索内存超限  | 160G建库内存超限  | 0.893228  | 0.893212  | 0.947941  |
| QPS (Recall = 0.90)  | 2922  | 2185  |   |   | 9860  | 9607  | 15804  |

QPS & 召回率

![deep-100M](../ann-benchmarks/results/deep-100M.png)



#### BIGANN-10M



|   | Faiss-IVF  | Faiss-IVFPQ  |  Faiss-HNSW | Nmslib(HNSW)  | Puck  | Puck-Flat  | Tinker  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Build Params  | IVF100000,Flat  | IVF100000,PQ96  | HNSW32,Flat, "efConstruction" : 640  | "M": 32, "efConstruction": 640, "indexThreadQty":32  | "C":3000, "F":3000, "FN":16, "N":96  |  "C":3000, "F":3000, "FN":16, "N":0 | "C":3000, "F":3000,"tinker_neighborhood":16,"tinker_construction":600  |
| Index Size(G)  | 4.86123  | 1.28495  | 6.11273  | 6.64465  | 1.48694  | 5.02498  | 6.20208  |
| Recall （QPS=1W）  | 0.857864  | 0.776675 | 采样QPS=1W的数据失败  | 0.952535  | 0.979541  | 0.983525  | 0.987338  |
| QPS (Recall = 0.95)  | 3899  | 1661  | 5468  | 10034  | 17285  | 16259  | 22031  |

QPS & 召回率

![bigann-10M](../ann-benchmarks/results/bigann-10M.png)





#### DEEP-10M



|   | Faiss-IVF  | Faiss-IVFPQ  |  Faiss-HNSW | Nmslib(HNSW)  | Puck  | Puck-Flat  | Tinker  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Build Params  | IVF31622,Flat  | IVF31622,PQ96  | HNSW16,Flat, "efConstruction" : 640  | "M": 16, "efConstruction": 640, "indexThreadQty":32  | "C":1000, "F":1000, "FN":16, "N":96  |  "C":1000, "F":1000, "FN":16, "N":0 | "C":1000, "F":1000, "tinker_neighborhood":16,"tinker_consstruction":600  |
| Index Size(G)  | 3.6652  | 0.983101  | 4.92067  | 5.45275  | 1.18808 | 3.83257  | 5.01369  |
| Recall （QPS=1W）  | 0.858472  | 0.776675 | 采样QPS=1W的数据失败  | 0.961981  | 0.979596  | 0.981993  | 0.989963  |
| QPS (Recall = 0.95)  | 6174  | 3574  | 5623  | 10708  | 17772  | 17110  | 25204  |

QPS & 召回率

![deep-10M](../ann-benchmarks/results/deep-10M.png)
