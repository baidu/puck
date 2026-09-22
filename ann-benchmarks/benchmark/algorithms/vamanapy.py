from benchmark.algorithms.base import BaseANN
from benchmark.algorithms.base import CPU_LIMIT
from benchmark.datasets import DATASETS, download_accelerated
import os
import struct
import time

import numpy as np
import vamanapy as vp

vp.set_num_threads(CPU_LIMIT)
class Vamana(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        print("Vamana: L_Build = " + str(self.l_build))
        print("Vamana: R = " + str(self.max_outdegree))
        print("Vamana: Alpha = " + str(self.alpha))
        self.params = vp.Parameters()
        self.params.set("L", self.l_build)
        self.params.set("R", self.max_outdegree)
        self.params.set("C", 750)
        self.params.set("alpha", self.alpha)
        self.params.set("saturate_graph", False)
        self.params.set("num_threads", 1)
        self.build_memory_usage = -1
    
    def fit(self, dataset):
        X = DATASETS[dataset]().get_dataset()
        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        print("Vamana: Starting Fit...")
        index_dir = "data/indices"

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        data_path = os.path.join(index_dir, "base.bin")
        self.name = "Vamana-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print("Vamana: Index Stored At: " + save_path)
        print(X.shape[0], X.shape[1])
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("Vamana: Creating Index")
            s = time.time()
            if self.metric == "l2":
                index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
            elif self.metric == "cosine":
                index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, data_path)
            else:
                print("Vamana: Unknown Metric Error!")
            index.build(self.params, [])
            t = time.time()
            print("Vamana: Index Build Time (sec) = " + str(t - s))
            index.save(save_path)
        if os.path.exists(save_path):
            print("Vamana: Loading Index: " + str(save_path))
            s = time.time()
            if self.metric == "l2":
                self.index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
            elif self.metric == "cosine":
                self.index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, data_path)
            else:
                print("Vamana: Unknown Metric Error!")
            self.index.load(file_name=save_path)
            print("Vamana: Index Loaded")
            self.index.optimize_graph()
            print("Vamana: Graph Optimization Completed")
            t = time.time()
            print("Vamana: Index Load Time (sec) = " + str(t - s))
        else:
            print("Vamana: Unexpected Index Build Time Error")

        print("Vamana: End of Fit")
    def load_index(self, dataset):
        self.fit(dataset)
       
    def set_query_arguments(self, l_search):
        print("Vamana: L_Search = " + str(l_search))
        self.l_search = l_search

    def query(self, X, n):
        self.num_queries = X.shape[0]
        self.result = np.ndarray(shape=(0,), dtype=int, order='F')
        # 批量检索接口使用dynamic openMP，并发度与物理机CPU核数有关，不受docker的cpu设置限制
        # vp.set_num_threads(CPU_LIMIT) 对结果无影响，当物理机CPU核数不等于CPU_LIMIT时，不能直接使用
        #self.result = self.index.batch_numpy_query(X, n, self.num_queries, self.l_search)
        for i0 in range(0, len(X), CPU_LIMIT) :
            real_step = min(CPU_LIMIT, len(X) - i0)
            Ii = self.index.batch_numpy_query(X[i0:i0 + real_step], n, real_step, self.l_search)
            self.result = np.hstack((self.result, Ii))


    def get_results(self):
        return self.result.reshape((self.num_queries, self.result.shape[0] // self.num_queries))


class VamanaPQ(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alpha = float(param["alpha"])
        self.chunks = int(param["chunks"])
        print("Vamana PQ: L_Build = " + str(self.l_build))
        print("Vamana PQ: R = " + str(self.max_outdegree))
        print("Vamana PQ: Alpha = " + str(self.alpha))
        print("Vamana PQ: Chunks = " + str(self.chunks))
        self.params = vp.Parameters()
        self.params.set("L", self.l_build)
        self.params.set("R", self.max_outdegree)
        self.params.set("C", 750)
        self.params.set("alpha", self.alpha)
        self.params.set("saturate_graph", False)
        self.params.set("num_chunks", self.chunks)
        self.params.set("num_threads", 1)
        self.build_memory_usage = -1

    def fit(self, dataset):
        X = DATASETS[dataset]().get_dataset()
        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        print("Vamana PQ: Starting Fit...")
        index_dir = "data/indices"

        if self.chunks > X.shape[1]:
            raise ValueError

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        data_path = os.path.join(index_dir, "base.bin")
        pq_path = os.path.join(index_dir, "pq_memory_index")
        self.name = "VamanaPQ-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alpha)
        save_path = os.path.join(index_dir, self.name)
        print("Vamana PQ: Index Stored At: " + save_path)
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("Vamana PQ: Creating Index")
            s = time.time()
            if self.metric == "l2":
                index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
            elif self.metric == "cosine":
                index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, data_path)
            else:
                print("Vamana PQ: Unknown Metric Error!")
            index.pq_build(data_path, pq_path, self.params)
            t = time.time()
            print("Vamana PQ: Index Build Time (sec) = " + str(t - s))
            index.save(save_path)
        if os.path.exists(save_path):
            print("Vamana PQ: Loading Index: " + str(save_path))
            s = time.time()
            if self.metric == "l2":
                self.index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
            elif self.metric == "cosine":
                self.index = vp.SinglePrecisionIndex(vp.Metric.INNER_PRODUCT, data_path)
            else:
                print("Vamana PQ: Unknown Metric Error!")
            self.index.load(file_name=save_path)
            print("Vamana PQ: Index Loaded")
            self.index.pq_load(pq_prefix_path=pq_path)
            print("Vamana PQ: PQ Data Loaded")
            self.index.optimize_graph()
            print("Vamana PQ: Graph Optimization Completed")
            t = time.time()
            print("Vamana PQ: Index Load Time (sec) = " + str(t - s))
        else:
            print("Vamana PQ: Unexpected Index Build Time Error")

        print("Vamana PQ: End of Fit")
    def load_index(self, dataset):
        self.fit(dataset)
       

    def set_query_arguments(self, l_search):
        print("Vamana PQ: L_Search = " + str(l_search))
        self.l_search = l_search

    def query(self, X, n):
        self.num_queries = X.shape[0]
        # 批量检索接口使用dynamic openMP，并发度与物理机CPU核数有关，不受docker的cpu设置限制
        # vp.set_num_threads(CPU_LIMIT) 对结果无影响，当物理机CPU核数不等于CPU_LIMIT时，不能直接使用
        #self.result = self.index.pq_batch_numpy_query(X, n, self.num_queries, self.l_search)
        self.result = np.ndarray(shape=(0,), dtype=int, order='F')
        for i0 in range(0, len(X), CPU_LIMIT) :
            real_step = min(CPU_LIMIT, len(X) - i0)
            Ii = self.index.pq_batch_numpy_quer(X[i0:i0 + real_step], n, real_step, self.l_search)
            self.result = np.hstack((self.result, Ii))


    def get_results(self):
        return self.result.reshape((self.num_queries, self.result.shape[0] // self.num_queries))
