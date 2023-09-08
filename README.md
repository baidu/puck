## Description
This project is a library for approximate nearest neighbor(ANN) search named Puck.
In Industrial deployment scenarios, limited memory, expensive computer resources and increasing database size are as important as the recall-vs-latency tradeof for all search applications.
Along with the rapid development of retrieval business service, it has the big demand for the highly recall-vs-latency and precious but finite resource, the borning of Puck is precisely for meeting this kind of need.

It contains two algorithms, Puck and Tinker. 
This project is written in C++ with wrappers for python3.  
Puck is an efficient approache for large-scale dataset, which has the best performance of multiple 1B-datasets in [NeurIPS'21 competition track](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips21/t1_t2/README.md#results-for-t1).
Since then, performance of Puck has increased by 70%. 
Puck includes a two-layered architectural design for inverted indices and a multi-level quantization on the dataset.
If the memory is going to be a bottleneck, Puck could resolve your problems.  
Tinker is an efficient approache for smaller dataset(like 10M, 100M), which has better performance than Nmslib in big-ann-benchmarks. 
The relationships among similarity points are well thought out, Tinker need more memory to save these. Thinker cost more memory then Puck, but has better performace than Puck. If you want a better searching performance and need not concerned about memory used, Tinker is a better choiese.

## Introduction

This project supports cosine similarity, L2(Euclidean) and IP(Inner Product, conditioned).
When two vectors are normalized, L2 distance is equal to 2 - 2 * cos.
IP2COS is a transform method that convert IP distance to cos distance.
The distance value in search result is always L2.  

Puck use a compressed vectors(after PQ) instead of the original vectors, the memory cost just over to 1/4 of the original vectors by default.
With the increase of datasize, Puck's advantage is more obvious.  
Tinker need save relationships of similarity points, the memory cost is more than the original vectors (less than Nmslib) by default.
More performance details in benchmarks. Please see [this readme](./ann-benchmarks/README.md) for more details.

## Linux install

### 1.The prerequisite is mkl, python and cmake.
**MKL**:  MKL must be installed to compile puck, download the MKL installation package corresponding to the operating system from the official website, and configure the corresponding installation path after the installation is complete.
source the MKL component environment script, eg. source ${INSTALL_PATH}/mkl/latest/env/vars.sh. This will maintain many sets of environment variables, like MKLROOT.

https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html

**python**: Version higher than 3.6.0.

**cmake**:  Version higher than 3.21.
### 2.Clone this project.
````shell
git clone https://github.com/baidu/puck.git
cd puck
````

### 3.Use cmake to build this project.
##### 3.1 Build this project
````shell
cmake -DCMAKE_BUILD_TYPE=Release 
    -DMKLROOT=${MKLROOT} \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DBLA_STATIC=ON  \
    -B build .

cd build && make && make install
````
##### 3.2 Build with GTEST 
Use conditional compilation variable named WITH_TESTING.
````shell
cmake -DCMAKE_BUILD_TYPE=Release 
    -DMKLROOT=${MKLROOT} \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DBLA_STATIC=ON  \
    -DWITH_TESTING=ON \
    -B build .

cd build && make && make install
````

##### 3.3 Build with Python

Refer to the [Dockerfile](./ann-benchmarks/install/Dockerfile.puck_inmem)
````shell
python3 setup.py install 
````

Output files are saved in build/output subdirectory by default.

## How to use
Output files include demos of train, build and search tools.  
Train and build tools are in build/output/build_tools subdirectory.  
Search demo tools are in build/output/bin subdirectory.

### 1.format vector dataset for train and build
The vectors are stored in raw little endian.
Each vector takes 4+d*4 bytes for .fvecs format, where d is the dimensionality of the vector.

### 2.train & build
The default train configuration file is "build/output/build_tools/conf/puck_train.conf".
The length of each feature vector must be set in train configuration file (feature_dim).

````shell
cd output/build_tools
cp YOUR_FEATURE_FILE puck_index/all_data.feat.bin
sh script/puck_train_control.sh -t -b
````

index files are saved in puck_index subdirectory by default.

### 3.search
During searching, the default value of index files path is './puck_index'.  
The format of query file, refer to [demo](./tools/demo/init-feature-example)  
Search parameters can be modified using a configuration file, refer to [demo](./demo/conf/puck.conf )

````shell
cd output/
ln -s build_tools/puck_index .
./bin/search_client YOUR_QUERY_FEATURE_FILE RECALL_FILE_NAME --flagfile=conf/puck.conf
````

recall results are stored in file RECALL_FILE_NAME.

## more details
[more details for puck](./docs/README.md)

## benchmark
Please see [this readme](./ann-benchmarks/README.md) for details.

this ann-benchmark is forked from https://github.com/harsha-simhadri/big-ann-benchmarks of 2021.

How to run this benchmark is the same with it. We add support of faiss(IVF,IVF-Flat,HNSW) , nmslib（HNSW）,Puck and Tinker of T1 track. And We update algos.yaml of these method using recommended parameters of 4 datasets(bigann-10M, bigann-100M, deep-10M, deep-100M)

## Discussion
Join our QQ group if you are interested in this project.

![QQ Group](./docs/PuckQQGroup.jpeg)
