# !/usr/bin/env python3
from __future__ import absolute_import
#import sys
#sys.path.append("install/lib-faiss")  # noqa
import numpy
import sklearn.preprocessing
import ctypes
import faiss
import time
import os
import math
import struct
from benchmark.algorithms.base import BaseANN
from benchmark.algorithms.base import CPU_LIMIT
from benchmark.datasets import DATASETS
from multiprocessing.pool import ThreadPool
import numpy as np

def knn_search_batched(index, xq, k, thread_limited):
    D, I = [], []
    
    for i0 in range(0, len(xq), thread_limited):
        real_step = min(thread_limited, len(xq) - i0)
        Di, Ii = index.search(xq[i0:i0 + real_step], k)
        D.append(Di)
        I.append(Ii)
    return numpy.vstack(D), numpy.vstack(I)


class Faiss(BaseANN):
    def query(self, X, n):
        if self._metric == 'angular':
            X /= numpy.linalg.norm(X)
        self.res = self.index.search(X.astype(numpy.float32), n)
        D, I = self.res
        self.I = I
    def get_results(self):
        return self.I
#        res = []
#        for i in range(len(D)):
#            r = []
#            for l, d in zip(L[i], D[i]):
#                if l != -1:
#                    r.append(l)
#            res.append(r)
#        return res
    def track(self):
        #T1 means in memory
        return "T1 for 10M & 100M"

class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric
        self.build_memory_usage = -1

    def index_name(self, name):
        return f"data/ivf_{name}_{self._n_list}_{self._metric}"

    def fit(self, dataset):
        X = DATASETS[dataset]().get_dataset() # assumes it fits into memory

        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        faiss.write_index(index, self.index_name(dataset))
        self.index = index

    def load_index(self, dataset):
        if not os.path.exists(self.index_name(dataset)):
            return False

        self.index = faiss.read_index(self.index_name(dataset))
        return True

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVF(n_list=%d, n_probe=%d)' % (self._n_list,
                                                    self._n_probe)

class FaissFactory(Faiss):
    def __init__(self, metric, index_params):
        self._index_params = index_params
        self._metric = metric
        self._query_bs = CPU_LIMIT
        self.indexkey = index_params.get("indexkey", "IVF4096")

        self.build_memory_usage = -1
        if 'query_bs' in index_params:
            self._query_bs = index_params['query_bs']

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.faissindex"

    def fit(self, dataset):
        index_params = self._index_params

        ds = DATASETS[dataset]()
        d = ds.d
        #IVF* & HNSW 建库过程中，使用OMP并行加速，并发度未指定，默认使用CPU核数
        # get build parameters
        metric_type = (
                faiss.METRIC_L2 if ds.distance() == "euclidean" else
                faiss.METRIC_INNER_PRODUCT if ds.distance() in ("ip", "angular") else
                1/0
        )
        index = faiss.index_factory(d, self.indexkey, metric_type)
        print(index.is_trained)
        index.verbose = True
        
        if isinstance(index, faiss.IndexHNSW) and "efConstruction" in index_params:
            index.hnsw.efConstruction = index_params.get("efConstruction")

        maxtrain = index_params.get("maxtrain", 0)
        if maxtrain == 0:
            maxtrain = max(maxtrain, 5000000)
            print("setting maxtrain to %d" % maxtrain)

        faiss.omp_set_num_threads(int(CPU_LIMIT))
        # train on dataset
        print(f"getting first {maxtrain} dataset vectors for training")
        if index.is_trained == False:
            xt2 = next(ds.get_dataset_iterator(bs=maxtrain))

            print("train, size", xt2.shape)
            assert numpy.all(numpy.isfinite(xt2))

            t0 = time.time()
            index.train(xt2)
            print("  Total train time %.3f s" % (time.time() - t0))
            
        print("adding")

        t0 = time.time()
        
        if ds.nb <= 10**7:
            index.add(ds.get_dataset())
        else:
            add_bs = index_params.get("add_bs", 10000000)
            i0 = 0
            for xblock in ds.get_dataset_iterator(bs=add_bs):
                i1 = i0 + len(xblock)
                print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                    i0, i1, ds.nb, time.time() - t0,
                    faiss.get_mem_usage_kb()))
                index.add(xblock)
                i0 = i1

        print("  add in %.3f s" % (time.time() - t0))
        print("storing", )
        faiss.write_index(index, self.index_name(dataset))

        self.index = index
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

    def load_index(self, dataset):
        print("Loading index")
        self.index = faiss.read_index(self.index_name(dataset))
        faiss.omp_set_num_threads(int(CPU_LIMIT))
        self.ps = faiss.ParameterSpace()
        self.ps.initialize(self.index)

        return True

    def set_query_arguments(self, query_args):
        faiss.cvar.indexIVF_stats.reset()
        self.ps.set_index_parameters(self.index, query_args)
        #print(self._query_bs) 
        self.qas = query_args

    def query(self, X, n):
        nq = X.shape[0]
        self.I = -np.ones((nq, n), dtype='int32')
        if self._query_bs == -1:
            D, self.I= self.index.search(X, n)
        else:
            def process_one_row(q):
                faiss.omp_set_num_threads(1)
                Di, Ii = self.index.search(X[q:q+1], n)
                self.I[q] = Ii
            
            faiss.omp_set_num_threads(self._query_bs)
            pool = ThreadPool(self._query_bs)
            list(pool.map(process_one_row, range(nq)))
            
    def __str__(self):
        return f'Faiss{self.indexkey, self.qas}'

###跑暴力用
class FaissIndexFlat(FaissFactory):
    def __init__(self, metric, index_params):
        self.fn_file = ""
        FaissFactory.__init__(self, metric, {'indexkey': 'Flat'})

    def fit(self, dataset):
        FaissFactory.fit(self, dataset)
        ds = DATASETS[dataset]()
        self.fn_file = ds.basedir + "/" + ds.gt_fn
    
    def get_results(self):
        D, I = self.res
        f = open(self.fn_file,'wb')
      
        n, d = D.shape
        buf = struct.pack('i', n)
        f.write(buf)
        buf = struct.pack('i', d)
        f.write(buf)

        W_I = I.flatten()
        N = W_I.shape
        buf = struct.pack('i' * N[0], *W_I)
        f.write(buf)

        D = D.flatten()
        N = D.shape
        buf = struct.pack('f' * N[0], *D)
        f.write(buf)
        f.close()
        
        return I
    def __str__(self):
        return f'Faiss{self.indexkey}'

