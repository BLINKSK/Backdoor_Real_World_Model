#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import ast
import logging
import argparse
from multiprocessing import Pool
from datetime import datetime

import collect
from utils import get_cmd

class modelInfer:

    """
    Rules:
    Rule 1: IF corpus.type == word THEN match the keyword list.
    Rule 2: IF corpus.type == phase, THEN split it into words, for each word in words_list, goto Rule 1.
    """

    task_list       = {
            'object detection'   : ['ssd', 'onet', 'rnet', 'pnet', 'detect'], 
            'pose detection'     : ['pose'], 
            'stylization'        : ['styl'],
            'sequence prediction': ['rnn', 'lstm'], 
            'classification'     : ['class', 'softmax', 'recog'],
            }

    optimize_list   = ['optimiz', 'quant']
    
    arch_list       = ["inceptionresnet", "resnet", "mobilenet", "vgg", 'mnas', 'squeeze', 'effient']

    input_type_list = {'image' : ['img', 'image', 'camera', 'picture'],
                       'audio' : ['audio', 'speeech', 'voice'],
                       'text'  : ['ocr', 'translate'],
                    }
    
    noun_list = ['detector', 'detection',
                 'classifier', 'classification',
                 'inference',
                 'predictor', 'prediction',
                 'recognizer', 'recognition',
                 #'machine learning', 'ml',
                 #'neural network', 'nn'
            ]
    output_label_list = ['enum', 'switch', 'case', 'result', 'output', 'label']
    preprocess_param  = ['image_mean', 'image_std', 'preprocess', '>> 8) & 255)']

    network_regex = "(\w+net)"
    entity_regex  = "(\w+)"
    

    def infer_model_task(self, m, target):
        
        find = False
        for task in self.task_list:
            for magic in self.task_list[task]:
                if magic in target:
                    m.task = task
                    find = True
                    break
            if find: break
        if find == False: m.task = 'unknown'
        return m

    def infer_model_quant(self, m, target):
        
        for i in self.optimize_list:
            if i in target:
                m.quant_level = 'quantized'
                break
        return m

    def infer_model_arch(self, m, target):
        """
        models with the same backbone are like a model family.
        """
        for i in self.arch_list:
            if i in target:
                m.backbone = i
                return m

        res = None
        if m.backbone == 'unknown' or m.backbone == '':
            res = re.search(self.network_regex, target)
        
        if res != None:
            #print(m.model_path, res.group())
            m.backbone = res.group()
        elif m.backbone == '':
            m.backbone = 'unknown'

        return m
    
    def infer_model_input_type(self, m, target=None, file_list=[], file_list_content=[]):

        if target != None:
            for input_type in self.input_type_list:
                for magic in self.input_type_list[input_type]:
                    if magic in target:
                        m.input_type += f"|{input_type}"
                        break
        
        if file_list != []:
            pass

        return m

    def infer_model_preprocess_param(self, m, file_list, file_list_content=[]):

        for content in file_list_content:
            for i in self.preprocess_param:
                if i in content:
                    m.preprocess_param += f"|Y"
                    break

        return m

    def infer_model_output_label(self, m, file_list, file_list_content=[]):

        for file_name in file_list:
            for noun in self.noun_list:
                res = re.findall(self.entity_regex + noun, file_name.decode('utf-8'), re.I)
                if res != []:
                    m.output_label += f'|{str(res)}'

        return m

# use regex to improve
black_list = [b"MANIFEST.MF", b"CERT.SF", b"GOOGPLAY.SF", b"apktool.yml"]

def decompose_src(args):
    src_name = args[1]
    smali_dst_name = f"{infer_args.DST_DIR}/smali_dec_output/{args[0]}.d/"
    if os.path.exists(smali_dst_name) == False:
        cmd = ["java", "-Djava.awt.headless=true", "-jar", infer_args.APKTOOL_NAME, "d", src_name, "--force", "-m", "--output", smali_dst_name]
        print(' '.join(cmd))
        retcode = get_cmd(cmd)

    java_dst_name = f"{infer_args.DST_DIR}/java_dec_output/{args[0]}.d/"
    if os.path.exists(java_dst_name) == False:
        cmd = [infer_args.JADX_NAME, "--deobf", "--deobf-use-sourcename", "--deobf-cfg-file-mode", "overwrite", "--show-bad-code", "--deobf-parse-kotlin-metadata", "-j", "8", "-d", java_dst_name, src_name]
        print(' '.join(cmd))
        retcode = get_cmd(cmd)
        print(retcode)

    # NOTE: whether this cmd succeeds or not, the flag 'd' is set.
    #if retcode == 0:
    return ('df', args[0]) # df means dec-full

def inference_test():
    global ai_app_num
    args = []
    ai_app = collect.query_db(infer_args.DB_NAME, "select hash, ori_path, format_res from ai_apk", is_raw=False)
    print(f"There are {len(ai_app)} ai apps.")
    logger.warning(ai_app)
    
    args = [(i[0], i[1]) for i in ai_app]
    if args == []:
        return
    logger.warning("Start dec-full...long time...")
    pool = Pool(processes=infer_args.CORE_NUM)
    map_result = pool.map(decompose_src, args)
    pool.close()
    pool.join()
    #collect.update_db(infer_args.DB_NAME, "update apk set feature=? where hash=?", map_result)
    logger.warning("Dec-full done!")
    
    form_list = [(i[0], i[2]) for i in ai_app if i[2] != '[]']
    for hash_, format_ in form_list:
        if do_smali_and_java_analysis(hash_, format_, infer_args.DST_DIR): ai_app_num += 1
        logger.warning("="*60)
    logger.warning(f"There are {ai_app_num} apps having inferenced their interfaces.")

def do_smali_and_java_analysis(hash_, format_, dst_dir):
    """
    @params
        hash_: apk hash
        format: 
        dst_dir: decompiled apk root path
    """
    smali_dec_full_path = f"{dst_dir}/smali_dec_output/{hash_}.d"
    java_dec_full_path = f"{dst_dir}/java_dec_output/{hash_}.d"
    
    format_l = ast.literal_eval(format_)
    print(format_l, type(format_l))
    if os.path.exists(smali_dec_full_path):
        for i in format_l:
            framework_name = i[0]
            path_l = i[1].split(b'\n')
            if path_l[-1] == b'':
                path_l = path_l[:-1]
            for path in path_l:
                pattern = b'.'.join(os.path.basename(path).split(b'.')[:-1]) # eliminate file suffix
                smali_res = find_smali_ref(smali_dec_full_path, pattern)
                
    if os.path.exists(java_dec_full_path):
        for i in format_l:
            framework_name = i[0]
            path_l = i[1].split(b'\n')
            if path_l[-1] == b'':
                path_l = path_l[:-1]
            for path in path_l:
                pattern = b'.'.join(os.path.basename(path).split(b'.')[:-1])
                java_ = find_java_ref(java_dec_full_path, pattern)
    return smali_res

def get_line(method_data):
    """
    from method of class get const value
    """
    patter_line = re.compile("[.line|.local*](.*?)(?=return|.line)", re.MULTILINE | re.DOTALL)
    data = re.findall(patter_line,method_data)
    return data

def get_called_methods(line):
    """
    gets all the method called inside a smali method data. works just fine with a single smali line
    :param content: content of the smali data to be parsed
    :rtype: list of lists
    :return: [0] - called method parameters
                [1] - called method object type
                [2] - called method name
                [3] - called method parameters object type
                [4] - called method return object type
    """
    pattern_called_methods = re.compile(r'invoke-.*?\ {(.*?)}, (.+?(?=;))\;\-\>(.+?(?=\())\((.*?)\)(.*?)(?=$|;)', re.MULTILINE | re.DOTALL)
    data = re.findall(pattern_called_methods, line)
    return data

def find_smali_ref(root_dir, model_name, cnt=True):
    """
    use rg to find references to these models in smali code.
    """
    res_flag = False
    cmd = ["rg", "-lie", model_name, root_dir]
    smali_call_point_path = get_cmd(cmd).split(b'\n')[:-1]
    
    for call_point in smali_call_point_path:
        if os.path.basename(call_point) not in black_list and call_point.split(b'/')[-2] != b"META-INF":
            logging.warning(f"[{model_name}]=> {call_point}")
            
            if call_point.endswith(b"smali") == False:
                logging.warning("NOT A SMALI FILE!")
                continue
            # find refs from call_points
            with open(call_point, 'r') as f:
                content = f.read()
                pattern_method_data = re.compile(r'^\.method.+?\ (.+?(?=\())\((.*?)\)(.*?$)(.*?(?=\.end\ method))', re.MULTILINE | re.DOTALL)
                methods = re.findall(pattern_method_data, content)
                if methods:
                    if cnt: res_flag = True
                    for method in methods:
                        logging.warning(f"method: {method[0]}\tarugment: {method[1]}\tretval: {method[2]}")
                else: 
                    logging.warning(f"None method have found!")
                    return

                for method in methods:
                    lines = get_line(method[3]) # get method body and split into lines.
                    for line in lines:
                        #print(line)
                        method_called = get_called_methods(line)
                        for m in method_called:
                            logging.warning(f"{method[0]} invokes: {m[2]}, invoked type: {m[1]}")
    return res_flag

def find_java_ref(root_dir, model_name):
    """
    use rg to find references to these java models.
    """
    cmd = ["rg", "-ie", model_name, root_dir]
    java_call_point_path = get_cmd(cmd).split(b'\n')[:-1]
    for item in java_call_point_path:
        logging.warning(item)

def find_java_cross_ref(root_dir, model_name):
    """
    use rg to find references to these java models.
    """
    cmd = ["rg", "-lie", model_name, root_dir]
    java_call_point_path = get_cmd(cmd).split(b'\n')[:-1]
    
    file_list = []
    file_list_content = []

    for item in java_call_point_path:
        if item.endswith(b"java") == True:
            file_list.append(item)
            with open(item, 'r') as f:
                content = f.read()
            file_list_content.append(content)

    return file_list, file_list_content

def infer_interface(m, log_folder):
    """
    return the input type, output labels and task description of model.
    """
    mi = modelInfer()

    smali_dec_full_path = log_folder + f"/decompile/smali_dec_output/{m.ori_apk_hash}.d"
    java_dec_full_path = log_folder + f"/decompile/java_dec_output/{m.ori_apk_hash}.d"
    print(smali_dec_full_path, java_dec_full_path)
    # NOTE: Trade-off
    # Here using model name without suffix as a pattern may bring too much FP.
    full_pattern = m.model_name
    pattern = '.'.join(full_pattern.split('.')[:-1])
    
    if os.path.exists(smali_dec_full_path):
        #find_smali_ref(smali_dec_full_path, pattern, cnt=False)
        pass
    if os.path.exists(java_dec_full_path):
        file_list, file_list_content = find_java_cross_ref(java_dec_full_path, pattern)
        # print('test_360/decompile_360')
        if file_list == []:
            # '2' means that we cannot find the MCP from code.
            m.model_type = '2'
        else:
            m = mi.infer_model_preprocess_param(m, file_list=file_list, file_list_content=file_list_content)
            m = mi.infer_model_output_label(m, file_list=file_list, file_list_content=file_list_content)
        #find_java_ref(java_dec_full_path, pattern)

    target = m.model_name.lower()
    
    m = mi.infer_model_task(m, target)
    
    # famous net
    m = mi.infer_model_arch(m, target)
    
    # inference model optimization
    m = mi.infer_model_quant(m, target)

    logging.warning("="*60+"infer DONE!")
    return m

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model interface inference.')
    parser.add_argument('--APKTOOL_NAME', help='Apktool path')
    parser.add_argument('--JADX_NAME', help='Jadx tool path')
    parser.add_argument('--DB_NAME', help='Src database path')
    parser.add_argument('--DST_DIR', help='Dst decompiled apk path')
    parser.add_argument('--CORE_NUM', type=int, help='parallel number')
    infer_args = parser.parse_args()
    
    ai_app_num = 0
    
    log_name = os.path.dirname(infer_args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')
    inference_test()
