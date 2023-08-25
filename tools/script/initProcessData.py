# -*- coding: UTF-8 -*-
#Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
@file: initProcessData.py
@author: yinjie06(yinjie06@baidu.com)
@date: 2018-04-22 15:57
@brief: init feature file according to puck_train.conf
"""
import struct
import sys
import time
import getopt
import re
import os
import math
import shutil
import fileinput
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')


def _usage():
    print '''
    py initProcessData.py [-f new feature file] [-h help]
    
    options:
        -f use for input file name
        -h use for help
    '''


class InitProcessData(object):
    """初始化."""

    def __init__(self, conf_file):
        """
          :param conf_file: 配置文件名
        
          初始化类：
        
          1）读取配置文件
          2）根据配置设置默认的输入路径和文件名称（如果没有指定则使用默认值），并打开相应的文件对象。
              - 如果指定的配置文件中有指定 `all_feature_file_name` 和 `keys_file_name`，将其作为相应的文件名读入；否则使用默认的文件名。
          3）设置是否归一化、IP2Cos计算特征等参数的值，并打印日志信息表示当前的参数设定值。
        """
        train_conf_file = conf_file
        train_conf_info = open(train_conf_file, 'rb').read()

        # 默认输入路径: mid-data,puck_index，如果不存在则创建
        if not os.path.exists('mid-data'):
            os.mkdir('mid-data')
        if not os.path.exists('puck_index'):
            os.mkdir('puck_index')

        # all_feature_file_name
        all_feature_file_name = "./puck_index/all_data.feat.bin" 
        conf_all_feature_file_name = re.findall(r'--feature_file_name=(\S+)', train_conf_info)
        if len(conf_all_feature_file_name) == 1:
            all_feature_file_name = conf_all_feature_file_name[0] 
        self.all_feature_file = open(all_feature_file_name, 'wb')
        print all_feature_file_name 
        
        # keys_file_name
        keys_file_name = "./puck_index/all_data.url"
        conf_keys_file_name = re.findall(r'--label_file_name=(\S+)', train_conf_info)
        if len(conf_keys_file_name) == 1:
            keys_file_name = conf_keys_file_name[0] 
        self.keys_file = open(keys_file_name, 'wb')
        print keys_file_name

        # whetherNorm
        self.whetherNorm = 1
        whetherNorm = re.findall(r'--whether_norm=(\S+)', train_conf_info)
        if len(whetherNorm) == 1:
            if whetherNorm[0] == '0' or whetherNorm[0] == 'false':
                self.whetherNorm = 0
        print self.whetherNorm

        self.ip2cos = 0
        ip2cos = re.findall(r'--ip2cos=(\d+)', train_conf_info)
        if len(ip2cos) == 1:
            self.ip2cos = int(ip2cos[0])
        if self.ip2cos == 1:
            self.whetherNorm = 0
        print self.ip2cos
            
        # feature_dim
        self.feature_dim = 256
        feature_dim = re.findall(r'--feature_dim=(\d+)', train_conf_info)
        if len(feature_dim) == 1:
            self.feature_dim = int(feature_dim[0])
        print self.feature_dim

    def init_process(self, feature_file):
        """格式化输出特征文件demo
            实时入库的索引需要每个样本有单独的label，
            其他情况非必须
        """
        """
        The vectors are stored in raw little endian. 
        Each vector takes sizeof(int)+dim * sizeof(float) bytes for .fvecs
        """
        all_line = 0
        valid_line = 0
        for line in fileinput.input(feature_file):
            all_line += 1
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
            
            feat = fields[-1].split(' ')
            if len(feat) != self.feature_dim:
                print "feature dim error, true dim = %d feature dim in conf = %d" % (len(feat), self.feature_dim) 
                return -1
            feat = np.array(map(float, feat))
            if (self.ip2cos > 1):
                return -1
            elif (self.ip2cos == 1):
                norm = np.dot(feat, feat)
                if norm > 1.0 or norm < 0.00001:
                    return -1
                feat = np.append(feat, math.sqrt(1.0 - norm))
            # 归一化
            elif(self.whetherNorm):
                if(np.sqrt(np.dot(feat, feat)) < 0.00001):
                    continue
                feat = feat / np.sqrt(np.dot(feat, feat))
            valid_line += 1
            self.keys_file.write(fields[0])
            self.keys_file.write('\n')
            
            buf = struct.pack('i', len(feat))
            self.all_feature_file.write(buf)
            buf = struct.pack('f' * len(feat), *feat)
            self.all_feature_file.write(buf)

        print 'all=%d, valid=%d' % (all_line, valid_line)
        self.all_feature_file.close()
        self.keys_file.close()
        return 0

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:f:h')
        if len(opts) < 1:
            _usage()
            sys.exit(1)
    except getopt.GetoptError as error:
        print 'params error! Error message: ' + error
        _usage()
        sys.exit(1)
    input_file = ''
    conf_file = 'conf/puck_index.conf'
    ret = 0 

    for opt, val in opts:
        print opt
        if opt == '-h':
            _usage()
            sys.exit(0)
        elif opt == '-i':
            if not os.path.exists(val):
                sys.exit(2)
            input_file = val
        elif opt == '-f':
            if not os.path.exists(val):
                sys.exit(2)
            conf_file = val
        else:
            _usage()
            sys.exit(1)

    init = InitProcessData(conf_file)
    ret = init.init_process(input_file)
    sys.exit(ret)
