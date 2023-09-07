## 关于Puck&Tinker

### Puck
&ensp;&ensp;&ensp;&ensp;名称来源：Puck源自经典MOBA游戏DOTA中的智力英雄，取其飘逸、灵动之意。<p>
&ensp;&ensp;&ensp;&ensp;ANN的检索性能是重中之重，Puck设计并实现了多种优化方案，着重提升性能和效果，包括但不限于：
* 采用二层倒排索引架构，能更敏感的感知数据分布，从而非常高效的分割子空间，减少搜索范围；同时采用共享二级类聚中心的方式，大幅减少训练时间
* 训练时采用启发式迭代的方法，不断优化一二级类聚中心，通过等价空间变换，训练获得更好的数据分布描述
* 采用多层级量化加速查找，优先通过大尺度量化的小特征快速找到候选集，再通过稍大一些的量化特征二次查找
* 在各个检索环节打磨极致的剪枝， 针对loss函数，通过多种公式变化，最大程度减少在线检索计算量，缩短计算时间
* 严格的内存cacheline对齐和紧致排列，最大程度降低cache miss
* 支持大尺度的量化，单实例支持尽可能多的数据，针对大尺度量化定向优化，减少量化损失; 同时支持非均匀量化，更加适应各种纬度的特征

&ensp;&ensp;&ensp;&ensp;除了性能以外，Puck还做了很多功能拓展：
* 实时插入：支持无锁结构的实时插入，做到数据的实时更新
* 条件查询：支持检索过程中的条件查询，从底层索引检索过程中就过滤掉不符合要求的结果，解决多路召回归并经常遇到的截断问题，更好满足组合检索的要求(暂未开源)
* 分布式建库：索引的构建过程支持分布式扩展，全量索引可以通过map-reduce一起建库，无需按分片build，大大加快和简化建库流程。  分布式建库工具(暂未开源)
* 自适应参数：ANN方法检索参数众多，应用起来有不小门槛，不了解技术细节的用户并不容易找到最优参数，Puck提供参数自适应功能，在大部分情况下使用默认参数即可得到很好效果

### Tinker
&ensp;&ensp;&ensp;&ensp;名称来源：Tinker同样源自经典MOBA游戏DOTA中的智力英雄<p> 
&ensp;&ensp;&ensp;&ensp;缘起：Puck在大数据集上表现优异，但在千万级以下的小数据集且要求高召回率的场景下优势减小（[benchmark](../ann-benchmarks/README.md)），我们思索说如何能继续突破，使得Puck在小数据集上性能更优<p> 
&ensp;&ensp;&ensp;&ensp;方案：经过不断尝试，我们提出了Tinker算法，Tinker的最终效果大大超出了最初预期,在benchmark上表现优异 

## 比赛获奖情况
&ensp;&ensp;&ensp;&ensp;首届国际向量检索大赛BigANN是由人工智能领域全球顶级学术会议NeurIPS发起，由微软、facebook等公司协办，是全球最高水平的赛事，旨在提升大规模ANN的研究创新和生产环境中的落地应用。虽是首届大赛，但因NeurIPS的极高知名度和权威性，吸引了众多知名企业和顶尖大学的同台竞技。本届比赛已于2021年12月NeurlPS’21会议期间公布结果  
&ensp;&ensp;&ensp;&ensp;Puck在参赛的四个数据集中均排名第一<p>
           比赛详情：https://big-ann-benchmarks.com/neurips21.html        
           比赛结果：https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips21/t1_t2/README.md#results-for-t1       


# 使用方法简介（训练、建库、检索、实时入库）
## 准备特征向量数据
### 1.特征文件  

&ensp;&ensp;&ensp;&ensp;特征文件是二进制（puck_index/all_data.feat.bin）。特征向量的纬度是dim，每个向量的存储格式是：sizeof(int)+dim * sizeof(float)。存储格式如下：
| field  | field type   |  description |
| :------------: | :------------: | :------------: |
| d  |  int | the vector dimension  |
|  components | float * d  | the vector components  |

### 2.标签文件  

&ensp;&ensp;&ensp;&ensp;标签文件（puck_index/all_data.url）是明文存储。特征向量在特征文件的顺序（local id），与其标签在标签文件的顺序保持一致。在实时插入、分布式建库等功能中，必须指定每个样本的标签。

### 3.数据格式化&校验工具

&ensp;&ensp;&ensp;&ensp;以上两个文件可通过工具自动生成。编译产出后，提供训练建库工具output/build_tools/script/puck_train_control.sh。

&ensp;&ensp;&ensp;&ensp;这个[工具](../tools/script/puck_train_control.sh)脚本输入数据格式如示例[文件](../tools/demo/init-feature-example), 用户可按示例格式准备数据，格式为key\tvector, vector中的每个float按空格分割。脚本对特征向量长度检查&预处理（归一、IP2COS等）后，写特征文件（puck_index/all_data.feat.bin）和标签文件（puck_index/all_data.url）。
````shell
cd output/build_tools
## 使用方法和查看help
sh script/puck_train_control.sh --help 
## 特征检查后，生成特征向量文件
sh script/puck_train_control.sh -i 特征文件

````

## 了解训练&建库&检索参数

&ensp;&ensp;&ensp;&ensp;该代码库的训练、建库和检索参数均通过gflags的方式指定。所有gflag定义参考[gflag](../puck/gflags/puck_gflags.cpp)。

### 核心训练&建库参数

#### 1.与数据集相关的参数

&ensp;&ensp;&ensp;&ensp;feature_dim：特征向量的纬度，由用户指定。

#### 2.影响训练效果的参数

##### 2.1检索算法的选择（index_type）

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;index_type = 1，指定检索算法类型为Puck；index_type = 2，指定检索算法类型为Tinker。

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Tinker有着绝对的性能优势，内存使用上大于Puck, 小于Nmslib；如果你的应用不是内存瓶颈，重点在计算上，请大胆使用Tinker。当在大规模数据集上，内存成为瓶颈时，Puck具备了更多优势，表现远优于Faiss-IVF和Faiss-IVFPQ，随着数据规模越大，Puck优势越明显。

<table align="center">
	<tr>
	    <td>算法</td>
	    <td>内存</td>
	    <td>检索性能</td>  
	    <td>训练参数</td>  
	</tr>
	<tr>
	    <td>Puck</td>
	    <td>内存消耗最少，<br>略高于样本总数 * dim * sizeof（char）</td>
	    <td rowspan="3">Tinker > Puck-Flat > Puck > HNSW(nmslib)> others</td>
	    <td>index_type=1</td>
	</tr>
	<tr>
	    <td>Puck-Flat</td>
	    <td>内存消耗比Puck高，<br>略高于样本总数 * dim * sizeof(float)</td>
	    <td>index_type=1<br>whether_pq=false</td>
	</tr>
	<tr>
	    <td>Tinker</td>
	    <td>三种方法中消耗最高，但低于HNSW（nmslib）</td>
	    <td>index_type=2</td>
	</tr>
</table>

##### 2.2聚类中心（coarse_cluster_count、fine_cluster_count）的选取与数据规模有关，推荐值如下：

<table align="center">
	<tr>
	    <td>数据规模</td>
	    <td>coarse_cluster_count</td>
	    <td>fine_cluster_count</td>  
	</tr>
	<tr>
	   <td> ≤500</td>
	    <td>500</td>
	    <td>500</td>  
	</tr>
    <tr>
	   <td>≤1kw</td>
	    <td>1000</td>
	    <td>1000</td>  
	</tr>
	<tr>
	   <td>≤5kw</td>
	    <td>2000</td>
	    <td>2000</td>  
	</tr>
	<tr>
	   <td>≤10kw</td>
	    <td>3000</td>
	    <td>3000</td>  
	</tr>
	<tr>
	   <td>＞10kw</td>
	    <td>5000</td>
	    <td>5000</td>  
	</tr>
</table>

##### 2.3其他

&ensp;&ensp;&ensp;&ensp;其他参数使用默认值，通常可以达到较好的效果。检索参数极限调优，可提issue讨论。  

&ensp;&ensp;&ensp;&ensp;影响训练效果的部分参数:

- *train_points_count*：训练聚类中心的样本个数，默认500w，从建库数据中随机抽样得到（抽样过程中会根据特征去重）。取值越大，训练需要内存越大，耗时越长。  
- *pq_train_points_count*：训练量化特征的样本个数，默认100w，从训练聚类中心的500w中随机抽取。取值越大，训练需要内存越大，耗时越长。根据经验，该值取值超过1kw后，量化误差差别不大。

&ensp;&ensp;&ensp;&ensp;Puck和Tinker有着不一样的索引结构，部分训练建库参数略有不同，如下。

&ensp;&ensp;&ensp;&ensp;**Puck & Puck-Flat:**
- *filter_nsq*：Puck使用2层量化，该参数指定第一层量化的量化比例，默认值 = feature_dim/4。取值越大，计算成本越高，量化误差越小。   
- *nsq*：该参数指定第二层量化的量化比例，默认值 = feature_dim。仅在whether_pq=true时，有效。 
- *whether_pq*：是否使用第二层量化，默认为true。当whether_pq=false时，该过程使用原始特征向量。

&ensp;&ensp;&ensp;&ensp;**Tinker**
- *tinker_neighborhood*：默认取值16。每个建库样本最多有tinker_neighborhood * 2 条边。每个样本存储图结构需要内存 = （2 * tinker_neighborhood + 1）* sizeof(int)。
- *tinker_construction*：默认取值600。建库过程中，邻居节点的候选集大小，通常tinker_construction >> tinker_neighborhood。建库过程中，根据三角形选边方法，从tinker_construction个候选集样本中，最多选取tinker_neighborhood * 2个样本作为最终的邻居。
### 核心检索参数
检索流程可以分为三个过程，每个过程都有自己的检索参数，如下：
#### 1.计算query与聚类中心的距离并排序  #### 

- *search_coarse_count*：检索一级聚类中心的个数，取值越大子空间的检索范围越大，需要<=coarse_cluster_count，一般默认值就足够

#### 2.计算query与top-M个聚类中心下样本的距离  #### 
&ensp;&ensp;&ensp;&ensp;Puck和Tinker有着不一样的索引结构，检索参数不同。

**Puck & Puck-Flat**:
- *filter_topk*：粗过滤候选集的大小，推荐调整范围 2~20倍的topK。取值越大，召回率越高，耗时增加，QPS下降。
- *radius_rate*：检索半径，与filter_topk配合使用，推荐调整范围1.0~1.05。检索范围越大，召回率越高，耗时增加，QPS下降。

**Tinker**

- *tinker_search_range*：结果集合队列长度。实际取值=std::max(tinker_search_range, topK)，检索结束时，返回该队列中的topK个样本。取值越大，检索范围越大，召回率越高，耗时增加，QPS下降。

#### 3.最终需要获取的TopK结果  #### 

- *topk* ： 默认100

## 训练 ##
&ensp;&ensp;&ensp;&ensp;编译产出后，训练建库工具在output/build_tools目录下。

&ensp;&ensp;&ensp;&ensp;当前支持本地训练。提供统一demo脚本（tools/script/puck_train_control.sh）。训练建库数据（puck_index/all_data.feat.bin）准备完成后，可直接执行。

````shell
##训练，在output/build_tools目录下
sh script/puck_train_control.sh -t
````
## 建库 ##

````shell
##建库，在output/build_tools目录下
sh script/puck_train_control.sh -b
````
## 检索 ##
&ensp;&ensp;&ensp;&ensp;创建索引的实例后，通过init()方法加载索引，search检索最近的topk个样本，获得相似度（distance）和样本的local idx（建库顺序）。response内的distance、local_idx生命周期自行维护。可参考[demo](./demo/search_client.cpp)文件。

&ensp;&ensp;&ensp;&ensp;检索API如下。

````c++
struct Request {
    uint32_t topk;              //检索时指定的最近topk参数，介于0到FLAGS_topk之间有效，其余值仍然以FLAGS_topk为准
    const float* feature;            //query feature
};
struct Response {
    float* distance;
    uint32_t* local_idx;
    uint32_t result_num;
};
/*
* @brief 检索最近的topk个doc
* @@param [in] request : search param
* @@param [out] response : search results
* @@return  0 => 正常, <0 => 错误
**/
int search(const Request* request, Response* response);
````
## 实时入库 ##
&ensp;&ensp;&ensp;&ensp;部分场景会需要实时插入数据功能，请创建RealtimeInsertPuckIndex，并调用insert方法。insert的数据先落盘再写入内存，当insert成功时保证数据磁盘和内存都写入成功。

&ensp;&ensp;&ensp;&ensp;可参考[demo](./demo/insert_demo.cpp)文件。

核心api如下：
````c++
/*
* @brief 实时插入 doc，线程安全
* @@param [in] insert_request : insert param
* @@param [out] log_string : 不为NULL 时返回debug日志信息
* @@return  0 => 正常, <0 => 错误
**/
int insert(const InsertRequest);
````

# 技术交流
QQ群（Puck技术交流群）：913964818

![QQ Group](PuckQQGroup.jpeg)

