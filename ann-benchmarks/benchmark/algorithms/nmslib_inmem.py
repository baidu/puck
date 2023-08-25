# !/usr/bin/env python3
from __future__ import absolute_import
import numpy as np
import sklearn.preprocessing
import ctypes
import os
import time
import nmslib
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS, download_accelerated

class NmslibHnsw(BaseANN):
    def __init__(self, metric, index_params):
        #josn格式
        self._index_params = index_params
        print(self._index_params)
        self._nmslib_metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]
        #默认值
        M = 16
        efC = 120
        key_list = sorted(self._index_params.keys())
        if 'M' in self._index_params.keys():
            M = self._index_params.get('M')
            key_list.remove('M')
        if 'efConstruction' in self._index_params.keys():
            efC = self._index_params.get('efConstruction')
            key_list.remove('efConstruction')
        if 'indexThreadQty' in self._index_params.keys():
            key_list.remove('indexThreadQty')
        self.indexkey = "M%s_efC%s"%(M, efC)
        for key in key_list:
            self.indexkey += "_%s%s"%(key, self._index_params.get(key))
        print(self.indexkey)
        self._save_index = True
        self._method_name = 'hnsw'
        print(self._nmslib_metric, self._method_name)
        self.build_memory_usage = -1
        self.index = nmslib.init(space=self._nmslib_metric, method=self._method_name)


    def track(self):
        #T1 means in memory
        return "T1 for 10M & 100M"

    def index_name(self, name):
        return f"data/{name}.{self.indexkey}.nmslibindex"

    def fit(self, dataset):
        #https://github.com/nmslib/nmslib/tree/9662fef7cb25bccd4431dd4d4e0bfc3a7c4927d7/python_bindings
        ds = DATASETS[dataset]()
        
        if ds.nb <= 10**7:
            self.index.addDataPointBatch(ds.get_dataset())
        else:
            add_part=100000
            i0 = 0
            t0 = time.time()
            for xblock in ds.get_dataset_iterator(bs=add_part):
                i1 = i0 + len(xblock)
                print("  adding %d:%d / %d [%.3f s] " % (i0, i1, ds.nb, time.time() - t0))
                ids = list(range(i0, i1))
                print(ids[0], ids[len(xblock)-1])
                self.index.addDataPointBatch(xblock, ids)
                i0 = i1
        any_params = []
        for key in self._index_params.keys():
            #建库不指定并发线程个数，默认使用CPU核数
            if key == "indexThreadQty":
                continue
            any_params.append("%s=%s"%(key, self._index_params[key]))
        print(any_params)
        self.index.createIndex(any_params, True)
        if self._save_index:
            self.index.saveIndex(self.index_name(dataset), save_data = True)
        print("hnsw build suc.")
        return True

    def load_index(self, dataset):
        if os.path.exists(self.index_name(dataset)):
            print('Loading index from file')
            self.index.loadIndex(self.index_name(dataset))
            return True
        return False

    def set_query_arguments(self, query_args):
        query_params = []
        query_params.append(query_args)
        #print(query_params)
        self.index.setQueryTimeParams(query_params)
        self.qas = "%s_%s"%(self.indexkey, query_args)

    def __str__(self):
        return f'Nmslib{self.indexkey, self.qas}'

    def query(self, X, n):
        #如果不指定限制的线程个数，会读取系统的cpu核数作为线程个数
        self.res = self.index.knnQueryBatch(X, n, self._index_params.get('indexThreadQty'))

    def get_results(self):   
        return [x for x, _ in self.res]
