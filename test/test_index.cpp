//Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.


/**
 * @file    test_index.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/04/13 16:18
 * @brief
 *
 **/

#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include "test/test_index.h"
#include "puck/gflags/puck_gflags.h"
#include "puck/hierarchical_cluster/hierarchical_cluster_index.h"
#include "puck/puck/puck_index.h"
#include "puck/tinker/tinker_index.h"
#include "puck/puck/realtime_insert_puck_index.h"
namespace puck {
DEFINE_string(index_data_url, "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz", "");
DEFINE_int32(test_data_dim, 128, "");
DEFINE_int32(query_cnt, 100, "");

int shell_cmd(std::string& sys_cmd_str) {
    char buff[1024];
    sprintf(buff, "%s", sys_cmd_str.c_str());

    FILE* fstream = nullptr;
    fstream = popen(buff, "r");

    if (!fstream) {
        return -1;
    }

    char tmp[1024];

    while (fgets(tmp, sizeof(tmp), fstream) != nullptr) {
        std::cout << tmp << std::endl; // can join each line as string
    }

    if (fstream) {
        pclose(fstream);
    }

    return 0;
}

int TestIndex::download_data() {
    std::string download_cmd = "rm -f siftsmall.tar.gz && rm -rf siftsmall && wget " + FLAGS_index_data_url +
                               " -O siftsmall.tar.gz";

    //std::string download_cmd = "rm -f siftsmall.tar.gz && cp ../../siftsmall.tar.gz .";
    if (shell_cmd(download_cmd) != 0) {
        return -1;
    }

    std::string train_data_cmd = "rm -rf " + FLAGS_index_path + " && mkdir " + FLAGS_index_path +
                                 " && tar zxvf siftsmall.tar.gz && cp siftsmall/siftsmall_base.fvecs " + FLAGS_index_path + "/" +
                                 FLAGS_feature_file_name;

    if (shell_cmd(train_data_cmd) != 0) {
        return -1;
    }

    std::string feature_dim = std::to_string(FLAGS_test_data_dim);
    google::SetCommandLineOption("feature_dim", feature_dim.c_str());
    google::SetCommandLineOption("whether_norm", "false");
    google::SetCommandLineOption("kmeans_iterations_count", "1");
    google::SetCommandLineOption("coarse_cluster_count", "100");
    google::SetCommandLineOption("fine_cluster_count", "100");
    _query_filename = "siftsmall/siftsmall_query.fvecs";
    _groundtruth_filename = "siftsmall/siftsmall_groundtruth.ivecs";
    return 0;
}

static int insert(RealtimeInsertPuckIndex* index, const float* feature, std::string lable) {
    InsertRequest insert_request;
    insert_request.feature = feature;
    insert_request.label = lable;
    LOG(INFO) << "insert " << lable;
    return index->insert(&insert_request);
}

int TestIndex::insert_index(int thread_cnt) {
    _index.reset(new puck::PuckIndex());

    if (_index->train() != 0) {
        LOG(ERROR) << "train Fail.\n";
        return -1;
    }

    puck::IndexConf conf = puck::load_index_conf_file();

    size_t batch_build_cnt = conf.total_point_count / 3;
    std::string cp_feature_data_cmd = "cp " + conf.feature_file_name + " " + conf.feature_file_name + "_bk";

    if (shell_cmd(cp_feature_data_cmd) != 0) {
        return -1;
    }

    std::vector<float> feature(conf.total_point_count * conf.feature_dim);

    int ret = read_fvec_format(conf.feature_file_name.c_str(), conf.feature_dim,
                               conf.total_point_count, feature.data());

    //_groundtruth_filename
    if (ret != conf.total_point_count) {
        LOG(ERROR) << "load " << conf.feature_file_name << " has Error";
        return -1;
    }


    std::ofstream lable_file(conf.label_file_name.c_str(), std::ios::binary | std::ios::out);
    uint32_t idx = 0;

    for (; idx < batch_build_cnt; ++idx) {
        std::string label = std::to_string(idx) + "\n";
        lable_file.write(label.c_str(), label.length());
    }

    lable_file.close();

    write_fvec_format(conf.feature_file_name.c_str(), conf.feature_dim, batch_build_cnt, feature.data());

    if (_index->build() != 0) {
        LOG(ERROR) << "build Fail.\n";
        return -1;
    }

    _index.release();

    std::unique_ptr<puck::RealtimeInsertPuckIndex> insert_index(new puck::RealtimeInsertPuckIndex());

    if (insert_index->init() != 0) {
        LOG(ERROR) << "RealtimeInsertPuckIndex init Fail.\n";
        return -1;
    }

    LOG(INFO) << "\n\nstart insert \n\n";
    std::vector<std::thread> writers;

    for (int i = conf.total_point_count - 1 ; i >= idx; --i) {
        std::string label = std::to_string(i);
        //insert(insert_index.get(), feature.data() + i * conf.feature_dim, label);
        //continue;
        std::thread write_index(insert, insert_index.get(), feature.data() + (uint64_t)i * conf.feature_dim, label);
        writers.push_back(std::move(write_index));

        if (writers.size() >= thread_cnt) {
            for (auto& t : writers) {
                t.join();
            }

            writers.clear();
        }
    }

    for (auto& t : writers) {
        t.join();
    }

    _index.reset(new puck::RealtimeInsertPuckIndex());
    return 0;
}

int TestIndex::build_index(int index_type) {
    if (index_type == int(puck::IndexType::TINKER)) {
        _index.reset(new puck::TinkerIndex());
    } else if (index_type == int(puck::IndexType::PUCK)) {
        _index.reset(new puck::PuckIndex());
    } else if (index_type == int(puck::IndexType::HIERARCHICAL_CLUSTER)) {
        _index.reset(new puck::HierarchicalClusterIndex());
    } else {
        LOG(ERROR) << "index type error.\n";
        return -1;
    }

    if (_index->train() != 0) {
        LOG(ERROR) << "train Fail.\n";
        return -1;
    }

    if (_index->build() != 0) {
        LOG(ERROR) << "build Fail.\n";
        return -1;
    }

    //_index.release();
    return 0;
}

float TestIndex::cmp_search_recall() {
    puck::IndexConf conf = puck::load_index_conf_file();
    /*
    if (conf.index_type == puck::IndexType::TINKER) { //Tinker
        LOG(INFO) << "init index of Tinker";
        _index.reset(new puck::TinkerIndex());
    } else if (conf.index_type == puck::IndexType::PUCK) {
        LOG(INFO) << "init index of Puck";
        _index.reset(new puck::PuckIndex());
    } else if (conf.index_type == puck::IndexType::HIERARCHICAL_CLUSTER) {
        LOG(INFO) << "init index of Flat";
        _index.reset(new puck::HierarchicalClusterIndex());
    } else {
        LOG(ERROR) << "init index of Error, Nan type";
        return -1;
    }*/



    if (_index->init() != 0) {
        LOG(ERROR) << "load index has Error";
        return -1;
    }

    LOG(INFO) << "load index suc.";

    std::vector<float> query_feature(FLAGS_query_cnt * conf.feature_dim);
    int ret = read_fvec_format(_query_filename.c_str(), conf.feature_dim,
                               FLAGS_query_cnt, query_feature.data());

    //_groundtruth_filename
    if (ret != FLAGS_query_cnt) {
        LOG(ERROR) << "load " << _query_filename << " has Error";
        return -1;
    }

    std::vector<std::vector<uint32_t>> groundtruth_data(FLAGS_query_cnt);
    {
        std::ifstream input_file;
        input_file.open(_groundtruth_filename.c_str(), std::ios::binary);

        if (!input_file.good()) {
            input_file.close();
            LOG(FATAL) << "read all data file error : " << _groundtruth_filename;
            return -1;
        }

        uint32_t d = 0;
        uint32_t i = 0;

        while (!input_file.eof() && i < FLAGS_query_cnt) {

            input_file.read((char*) &d, sizeof(uint32_t));
            //LOG(INFO)<<_groundtruth_filename<<" dim = "<<d<<" "<<i;
            groundtruth_data[i].resize(d);
            input_file.read((char*)groundtruth_data[i].data(), sizeof(uint32_t) * d);
            ++i;
        }

        input_file.close();
    }

    LOG(INFO) << 1;
    Request request;
    Response response;
    request.topk = FLAGS_topk;

    std::vector<float> distance(request.topk * FLAGS_query_cnt);
    std::vector<uint32_t> local_idx(request.topk * FLAGS_query_cnt);
    response.distance = distance.data();
    response.local_idx = local_idx.data();
    uint32_t match_pair_cnt = 0;

    RealtimeInsertPuckIndex* insert_index = dynamic_cast<RealtimeInsertPuckIndex*>(_index.get());

    for (int i = 0; i < FLAGS_query_cnt; ++i) {
        request.feature = query_feature.data() + i * conf.feature_dim;
        ret = _index->search(&request, &response);

        if (ret != 0) {
            LOG(ERROR) << "search item " << i << " error" << ret;
            break;
        }

        if (insert_index != nullptr) {
            for (int j = 0; j < (int)response.result_num; j ++) {
                std::string lable;

                if (insert_index->get_label(response.local_idx[j], lable) != 0) {
                    LOG(ERROR) << "get_label item " << response.local_idx[j] << " error";
                    break;
                }

                //LOG(INFO)<<response.local_idx[j]<<"\t"<<lable;
                response.local_idx[j] = std::atoi(lable.c_str());
            }
        }

        for (int j = 0; j < (int)response.result_num; j ++) {
            //LOG(INFO)<<"\t"<<i<<"\t"<<j<<"\t"<<response.local_idx[j]<<" "<<groundtruth_data[i][j];

            auto ite = std::find(groundtruth_data[i].begin(), groundtruth_data[i].end(), response.local_idx[j]);

            if (ite != groundtruth_data[i].end()) {
                ++match_pair_cnt;
            }
        }
    }

    LOG(INFO) << "match_pair_cnt=" << 1.0 * match_pair_cnt / (FLAGS_query_cnt * FLAGS_topk);;
    return 1.0 * match_pair_cnt / (FLAGS_query_cnt * FLAGS_topk);
}

}//namespace puck
