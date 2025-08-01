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

def decompose_no_src(args):
    src_name = args[1]
    # one app may have the same name, but have different versions.
    dst_name = f"{dec_args.DST_DIR}/{args[0]}.d"
    if os.path.exists(dst_name) == False:
        cmd = ["java", "-Djava.awt.headless=true", "-jar", dec_args.APKTOOL_NAME, "d", src_name, "--no-src", "--force", "-m", "--output", dst_name]
        get_cmd(cmd)
    # NOTE: whether this cmd succeeds or not, the flag 'd' is set.
    return ('d', args[0])
    
def decompose_test():
    query_result = collect.query_db(dec_args.DB_NAME, "select hash, path from apk where (feature!='d') or (feature is null)", is_raw=False)
    logger.warning(query_result)
    args = []
    for i in query_result:
        if os.path.exists(i[1]):
            args.append(i)
    if args == []:
        return
    pool = Pool(processes=dec_args.CORE_NUM)
    map_result = pool.map(decompose_no_src, args)
    pool.close()
    pool.join()
    
    logger.debug(map_result)
    collect.update_db(dec_args.DB_NAME, "update apk set feature=? where hash=?", map_result)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Coarse-grained decompose apk files in parallel.')
    parser.add_argument('--APKTOOL_NAME', help='Apktool path')
    parser.add_argument('--DB_NAME', help='Src database path')
    parser.add_argument('--DST_DIR', help='Dst decomposed data path')
    parser.add_argument('--CORE_NUM', type=int, help='parallel number')
    dec_args = parser.parse_args()

    log_name = os.path.dirname(dec_args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')
    
    decompose_test()
