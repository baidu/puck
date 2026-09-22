#!/bin/bash
#########################################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
# @file	swig_run.sh
# @author	yinjie06(yinjie06@baidu.com)
# @date	2021-08-18 14:52
# @brief	
#########################################################################

# SWIG 可执行文件，可通过环境变量 SWIG 覆盖，默认从 PATH 查找
SWIG_BIN="${SWIG:-swig}"

mkdir -p py_puck_api
"$SWIG_BIN" -c++ -python -outdir py_puck_api -o ./py_puck_api/py_puck_api.cpp ./pyapi_wrapper/py_puck_api.i
