import os
import re
import ast
import sys
import logging
import argparse
from shutil import copyfile
from datetime import datetime

import collect
from utils import gen_hash

def save_model(app_hash, model_type, ori_model_path, filename):
    dst_path = os.path.join(ext_args.MODEL_DIR, model_type, app_hash)
    dst_name = os.path.join(dst_path, filename)
    if os.path.exists(dst_path) == False:
        os.makedirs(dst_path)
    hash_ = gen_hash(ori_model_path)
    copyfile(ori_model_path, dst_name+'_'+hash_)

def extract_test():
    ai_app = collect.query_db(ext_args.DB_NAME, "select hash,format_res from ai_apk", is_raw=False)
    form_list = [(i[0], i[1]) for i in ai_app]
    for hash_, format_ in form_list:
        dec_path = f"{ext_args.DEC_DIR}/{hash_}.d"
        if os.path.exists(dec_path):
            types = ast.literal_eval(format_)
            for t in types:
                for i in t[1].split(b'\n')[:-1]:
                    filename = os.path.basename(i.decode())
                    if re.match(black_list, filename) == None:
                        logger.warning(f"model type: {t[0]}, model path: {i}")
                        save_model(hash_, t[0], i, filename)
            #cmd = "update ai_apk "
            #collect.update_db(cmd, hash_, smali_)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract models from apk components.')
    parser.add_argument('--DB_NAME', help='Src database path')
    parser.add_argument('--DEC_DIR', help='Src decomposed data path')
    parser.add_argument('--MODEL_DIR', help='Dst model save path')
    ext_args = parser.parse_args()

    black_list = '(?:%s)' % '|'.join(["feat\.params", "METADATA\.pb", "MANIFEST\.MF", "CERT\.SF", "GOOGPLAY\.SF", "apktool\.yml", ".*\.so$", ".*\.dex$"])

    log_name = os.path.dirname(ext_args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')
    
    extract_test()
