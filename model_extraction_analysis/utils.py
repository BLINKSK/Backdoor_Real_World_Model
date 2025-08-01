#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from hashlib import sha1
from subprocess import Popen, PIPE, run

import aaptlib
from item import apk_item

logger = logging.getLogger('dev.utils')
logger.setLevel(logging.WARNING)

def run_cmd(cmd):
    ret = run(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    if ret.returncode == 0:
        logger.error(f"[CMD_SUCC] {cmd}")
    else:
        logger.error(f"[CMD_ERR] {cmd} {ret.stderr}")
    return ret.returncode

def get_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=False)
    try:
        output = ''
        output, stderr = p.communicate()
    finally:
        if stderr:
            logger.error(f"[ERROR] Popen: {stderr}")
        #return output.splitlines()
        return output

def gen_hash(f):
    # note that if f has spaces
    shasum = get_cmd(['shasum', f])
    logger.debug(f"[DEBUG] {shasum.decode()}")
    return shasum.decode().split(' ')[0]

def parse_apk(path):
    '''
    @param path: file path
    @return an apk item.
    '''
    cur_apk = aaptlib.ApkInfo(apk_path=path)
    
    ret = apk_item()
    try:
        ret.apk_name = cur_apk.getPackage()
        ret.apk_path = path
        if ret.apk_name:
            #str_dump = cur_apk.getDumpStrings()
            #logger.warn("[STR] "+str(str_dump))
            ret.apk_hash = gen_hash(ret.apk_path)
            #dumb = a.getDumpResources() # NOTE: maybe very very slow
        
    except aaptlib.ApkInvalidError:
        logger.error(f"[ERROR] {path} Invalid apk!")
        return None
    return ret._tuple()
