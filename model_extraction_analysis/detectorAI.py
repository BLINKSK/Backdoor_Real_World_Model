#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
from multiprocessing import Pool
from datetime import datetime

import collect
from utils import get_cmd

def model_format(root_dir):
    suffix = [("TensorFlow", "\.pb$|\.pbtxt$|ckpt"),
            ("TensorflowLite", "\.tflite$|\.lite$"),
            ("Sensory", "libsmma"),
            ("Caffe", "\.caffemodel$|\.prototxt$"),
            ("Onfido", "onfido_liveness"),
            ("Jumio", "nv_liveness"),
            ("Megvii", "liblivenessdetection"),
            ("NCNN", "\.param$"),
            ("MxNet", "\.params$"),
            ("Xiaomi", "libmace"),
            ("Huawei", "libhiai"),
            ("Baidu", "libpaddle"),
            ("Alibaba", "\.mnn$"),
            ("SenseTime", "libst_mobile"),
            #("unknown", "\.model$")
            ]
    res = []
    _format = []
    for i,j in suffix:
        cmd = ["fd", "-uu", "--type", "f", j, root_dir]
        result = get_cmd(cmd)
        if result:
            res.append((i, result))
            _format.append(i)
    return (_format, res)

def magic_search(root_dir):
    magic = [("TensorFlow", "tensorflow"),
            ("TensorflowLite", "tflite"),
            ("Caffe", "caffe::Net"),
            ("Facebook", "caffe2|N6caffe26NetDefE"),
            ("Onfido", "onfido_liveness"),
            ("Jumio", "nv_liveness"),
            ("Megvii", "liblivenessdetection"),
            ("deeplearning4j", "Nd4jCpu"),
            ("SNPE", "snpe_get_tensor_dims"),
            ("MxNet", "N5mxnet6EngineE"),
            ("Tencent", "feathercnn"),
            ("Xiaomi", "libmace"),
            ("ByteDance", "bytenn"),
            #("SenseTime", "stmobile"),
            ("Alibaba", "MNNNet"),
            #("unknown", "cnn|dnn|lstm|neural network|tensor")
            ]
    res = []
    _magic = []
    for i,j in magic:
        # a => include bins, l => only show pathname, i => ignore letter cases, e => use regex
        cmd = ["rg", "-alie", j, root_dir]
        result = get_cmd(cmd)
        if result:
            res.append((i, result))
            _magic.append(i)
    return (_magic, res)

def ai_filter(args):
    # init
    flag = False
    target_dir = args[3]
    
    _format,_format_res = model_format(target_dir)
    if _format != []:
        flag = True
    _magic, _magic_res = magic_search(target_dir)
    if _magic != []:
        flag = True
    
    # for efficiency
    if not flag: return [False, None] 
    
    _frameworks = list(set(_format) | set(_magic))
    res = [flag, (args[0], args[1], args[2], args[3], str(_frameworks), str(_format_res), str(_magic_res))]
    #logger.debug(f"[DEBUG] single res {target_dir} => {str(res)}")
    return res

def detector_test():
    """
    framework : [framework_name, ...]
    format_res: [(framework_name, path\npath\n...), ...]
    magic_res : [(framework_name, path\npath\n...), ...]
    """
    ai_apk_schema = "create table if not exists ai_apk \
                            (hash text primary key,\
                            name text NOT NULL,\
                            ori_path text NOT NULL,\
                            dec_path text NOT NULL,\
                            framework text NOT NULL,\
                            format_res text NULL,\
                            magic_res text NULL\
                            )"
    collect.create_db(det_args.DB_NAME, ai_apk_schema)
    # not only once query DB with decomposed and tmp-non-AI apks. 
    query_result = collect.query_db(det_args.DB_NAME, "select hash,name,path from apk where feature='d'", is_raw=False) # and label=False
    logger.debug(f"[WARN] query result: {str(query_result)}")
    args = []
    for _hash,_name,_path in query_result:
        dst_dir = f"{det_args.DEC_DIR}/{_hash}.d"
        if os.path.exists(dst_dir) and os.path.isdir(dst_dir) and os.listdir(dst_dir):
            args.append((_hash, _name, _path, dst_dir))
    if args == []:
        return 0
    pool = Pool(processes=det_args.CORE_NUM)
    map_result = pool.map(ai_filter, args)
    pool.close()
    pool.join()
    
    update_apk_result    = []
    insert_ai_apk_result = []
    
    for i in map_result:
        if i[0] == True:
            update_apk_result.append((True, i[1][0]))
            insert_ai_apk_result.append(i[1])
    logger.warning(f"[WARN] find {len(update_apk_result)} AI apks.")
    if len(update_apk_result) > 0:
        collect.update_db(det_args.DB_NAME, "update apk set label=? where hash=?", update_apk_result)
        collect.insert_db(det_args.DB_NAME, "insert or replace into ai_apk values (?,?,?,?,?,?,?)", insert_ai_apk_result)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Detect models in apk components and store them into database.')
    parser.add_argument('--DB_NAME', help='Dst database path')
    parser.add_argument('--DEC_DIR', help='Src decomposed data path')
    parser.add_argument('--CORE_NUM', type=int, help='parallel number')
    det_args = parser.parse_args()
    
    log_name = os.path.dirname(det_args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')
    detector_test()
