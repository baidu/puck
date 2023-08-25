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
 * @file string_split.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2023/1/31 19:51
 * @brief
 *
 **/
#pragma once
#include <string>
#include <vector>
namespace puck{

u_int32_t s_split(const std::string& input_stream, const std::string& pattern, std::vector<std::string>& ret) {
    
    //在字符串末尾也加入分隔符，方便截取最后一段
    std::string strs = input_stream + pattern;
    size_t pos = strs.find(pattern);
    ret.clear();
    while(pos != strs.npos)
    {
        std::string temp = strs.substr(0, pos);
        if(temp.length() > 0){
            ret.push_back(temp);
        }else{
            break;
        }
        //去掉已分割的字符串,在剩下的字符串中进行分割
        strs = strs.substr(pos+1, strs.size());
        pos = strs.find(pattern);
    }
    //LOG(INFO)<<"s_split = "<<ret.size();
    return ret.size();
}
}

