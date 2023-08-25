#-*- coding:utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
@file: scann_inmem.py
@author: yinjie06(yinjie06@baidu.com)
@date: 2022-05-07 21:08
@brief: 
"""
from benchmark.algorithms.base import BaseANN
from benchmark.algorithms.base import CPU_LIMIT
from benchmark.datasets import DATASETS, download_accelerated
import scann
import numpy as np

#scann的参数选用的是开源库ann-benchmarks中scann在L2距离下的所有配置（https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml）
#scann的训练参数众多，配置中的参数不是最优的,欢迎提交新的配置更新benchmarkk的数据
class Scann(BaseANN):
    def __init__(self, metric, index_params):
        self._index_params = index_params
        print(self._index_params)
        self._metric = metric
        indexkey = index_params.get("indexkey", "NA")
        print("indexkey="%(indexkey))
        self.n_leaves = indexkey[0][0]
        self.avq_threshold = indexkey[1][0]
        self.dims_per_block = indexkey[2][0]
        self.dist = indexkey[3][0]
        self.build_memory_usage = -1

        self.name = "scann n_leaves={} avq_threshold={:.02f} dims_per_block={}".format(
            self.n_leaves, self.avq_threshold, self.dims_per_block)
        print(self.name)
        self.indexkey="n_leaves{}-avq_threshold={:.02f}-dims_per_block={}".format(
            self.n_leaves, self.avq_threshold, self.dims_per_block)
        self.topk = 10
        self.n = 0

    def track(self):
        #T1 means in memory
        return "T1 for 10M & 100M"
     
    def fit(self, dataset):
        X = DATASETS[dataset]().get_dataset()
        if self.dist == "dot_product":
            spherical = True
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            spherical = False
        print(spherical)
        self.searcher = scann.scann_ops_pybind.builder(X, 10, self.dist).tree(
            self.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True).score_ah(
                self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold).reorder(
                    1).build()
        #self.searcher.serialize_to_module()
        self.topk = DATASETS[dataset]().default_count()
        print("sss")

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.scannindex"

    def index_tag_name(self, name):
        return f"{name}.{self.indexkey}.scannindex"

    def load_index(self, dataset):
        return False
    
    def set_query_arguments(self, query_args):
        print(query_args)
        self.leaves_to_search, self.reorder = query_args
        print(self.leaves_to_search)
        print(self.reorder)
        self.qas=f"{self.leaves_to_search}_{self.reorder}"

    def query(self, X, n):
        #n, d = X.shape
        #这个函数是批量计算，实际计算过程中，是一个个的计算，无并发
        #self.res = self.searcher.search_batched(X, final_num_neighbors=self.topk,leaves_to_search=self.leaves_to_search, pre_reorder_num_neighbors=self.reorder)
        #这个函数是并行计算，读取系统CPU核数作为并发线程个数，对query分块后，块之间并行计算
        #self.res = self.searcher.search_batched_parallel(X, final_num_neighbors=self.topk,leaves_to_search=self.leaves_to_search, pre_reorder_num_neighbors=self.reorder)
        #return True
        #
        D, I = [], []

        for i0 in range(0, len(X), CPU_LIMIT):
            real_step = min(CPU_LIMIT, len(X) - i0)
            Ii, Di = self.searcher.search_batched_parallel(X[i0:i0 + real_step], final_num_neighbors=self.topk,leaves_to_search=self.leaves_to_search, pre_reorder_num_neighbors=self.reorder)
            D.append(Di)
            I.append(Ii)

        self.res = np.vstack(I), np.vstack(D)
        #print(self.res[0].shape)


    def get_results(self):
        return self.res[0]
    
    def __str__(self):
        return f'Scann{self.indexkey, self.qas}'
