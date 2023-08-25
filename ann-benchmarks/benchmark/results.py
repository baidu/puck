from __future__ import absolute_import

import h5py
import json
import os
import re
import traceback
import numpy


def get_result_filename(dataset=None, count=None, definition=None,
                        query_arguments=None):
    d = ['results']
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    if definition:
        d.append(definition.algorithm)
        build_args = definition.arguments
        try:
            for args in build_args:
                if type(args) == dict and 'indexkey' in args:
                    build_args = [args['indexkey']]
        except:
                pass
        data = build_args + query_arguments
        data = re.sub(r'\W+', '_', json.dumps(data, sort_keys=True)).strip('_')
        if len(data) > 150:
            data = data[-149:]
        d.append(data)

    return os.path.join(*d)


def store_results(dataset, count, definition, query_arguments,
        attrs, results, search_type):
    fn = get_result_filename(
        dataset, count, definition, query_arguments) + '.hdf5'
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    if search_type == "knn":
        neighbors = f.create_dataset('neighbors', (len(results), count), 'i')
        for i, idxs in enumerate(results):
            #print(len(idxs), type(idxs), idxs.shape)
            #部分算法存在如下情况：某个检索参数下，个别query的召回个数<count，补齐 
            if idxs.shape[0] < neighbors[i].shape[0]:
                id_list = idxs.tolist()
                for i in range(idxs.shape[0], neighbors[i].shape[0]):
                    id_list.append(0 - i)
                idxs = numpy.array(id_list)
                #print(idxs)
            neighbors[i] = idxs
    elif search_type == "range":
        lims, D, I= results
        f.create_dataset('neighbors', data=I)
        f.create_dataset('lims', data=lims)
        f.create_dataset('distances', data=D)
    else:
        raise NotImplementedError()
    f.close()


def load_all_results(dataset=None, count=None):
    """
    A generator for all result files.
    """
    for root, _, files in os.walk(get_result_filename(dataset, count)):
        for fn in files:
            if os.path.splitext(fn)[-1] != '.hdf5':
                continue
            try:
                f = h5py.File(os.path.join(root, fn), 'r+')
                properties = dict(f.attrs)
                yield properties, f
                f.close()
            except:
                print('Was unable to read', fn)
                traceback.print_exc()


def get_unique_algorithms():
    return set(properties['algo'] for properties, _ in load_all_results())
