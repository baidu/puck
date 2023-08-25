/***********************************************************************
 * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
 * @file    search_client.cpp
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2022-10-09 15:24
 * @brief
 ***********************************************************************/
#include <glog/logging.h>
#include "puck/puck/puck_index.h"
#include "puck/tinker/tinker_index.h"
#include "tools/string_split.h"

int read_feature_data(std::string& input_file, std::vector<std::string>& pic_name,
                      std::vector<std::vector<float> >& doc_feature) {
    std::ifstream fin;
    fin.open(input_file.c_str(), std::ios::binary);

    if (!fin.good()) {
        LOG(ERROR) << "cann't open output file:" << input_file.c_str();
        return -1;
    }

    int ret = 0;
    std::string line;

    pic_name.clear();
    doc_feature.clear();

    while (std::getline(fin, line)) {
        std::vector<std::string> split_str;

        if (puck::s_split(line, "\t", split_str) < 2) {
            LOG(ERROR) << "id:" << pic_name.size() << " get name error.";
            ret = -1;
            break;
        }

        pic_name.push_back(split_str[0]);
        std::string feature_str = split_str[1];

        puck::s_split(feature_str, " ", split_str);

        std::vector<float> cur_feature;

        for (u_int32_t idx = 0; idx < split_str.size(); ++idx) {
            cur_feature.push_back(std::atof(split_str[idx].c_str()));
        }

        doc_feature.push_back(cur_feature);
    }

    fin.close();
    LOG(INFO) << "total query cnt = " << pic_name.size();
    return ret;
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    //1. load index
    std::unique_ptr<puck::Index> index;
    puck::IndexType index_type = puck::load_index_type();

    if (index_type == puck::IndexType::TINKER) { //Tinker
        LOG(INFO) << "init index of Tinker";
        index.reset(new puck::TinkerIndex());
    } else if (index_type == puck::IndexType::PUCK) {
        LOG(INFO) << "init index of Puck";
        index.reset(new puck::PuckIndex());
    } else if (index_type == puck::IndexType::HIERARCHICAL_CLUSTER) {
        LOG(INFO) << "init index of Flat";
        index.reset(new puck::HierarchicalClusterIndex());
    } else {
        LOG(INFO) << "init index of Error, Nan type";
        return -1;
    }

    if (index == nullptr) {
        LOG(ERROR) << "create new SearchInterface error.";
        return -2;
    }

    int ret = index->init();

    if (ret != 0) {
        LOG(ERROR) << "SearchInterface init error " << ret;
        return -3;
    }
    //2. read input
    std::string input(argv[1]);
    std::string output(argv[2]);
    std::vector<std::vector<float> > in_data;
    std::vector<std::string> pic_name;

    ret = read_feature_data(input, pic_name, in_data);

    if (ret != 0) {
        LOG(ERROR) << "read_feature_data error:" << ret;
        return -4;
    } else {
        LOG(INFO) << "read_feature_data item:" << in_data.size();
    }

    //3. search
    const int item_count = in_data.size();
    FILE* pf = fopen(output.c_str(), "w");

    if (nullptr == pf) {
        LOG(ERROR) << "open outfile[" << output.c_str() << "] error.";
        return -1;
    }

    char buff[1024] = {0};
    puck::Request request;
    puck::Response response;
    request.topk = 100;

    response.distance = new float[request.topk];
    response.local_idx = new uint32_t[request.topk];

    for (int i = 0; i < item_count; ++i) {
        request.feature = in_data[i].data();

        ret = index->search(&request, &response);

        if (ret != 0) {
            LOG(ERROR) << "search item " << i << " error" << ret;
            break;
        }

        for (int j = 0; j < (int)response.result_num; j ++) {
            char* p = buff;
            //std::string lable = index->get_label(response.local_idx[j]);
            snprintf(p, 1024, "%s\t%d\t%f", pic_name[i].c_str(),
                     //lable.c_str(),
                     response.local_idx[j],
                     response.distance[j]);

            fprintf(pf, "%s\n", buff);
        }
    }

    delete [] response.distance;
    delete [] response.local_idx;
    fclose(pf);

    return 0;
}
