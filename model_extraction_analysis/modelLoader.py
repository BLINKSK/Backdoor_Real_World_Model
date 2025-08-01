# -*- coding: utf-8 -*-
# workon cleverhans
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import logging
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf
import collect
from item import model_item
from interfaceInference import infer_interface, modelInfer


def tflite_loader(path):
    #print("TFLite", path)
    try:
        with tf.device("/cpu:0"):
            # Load TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            #print("[INPUT]", input_details)
            #print("[OUTPUT]", output_details)
            details = interpreter.get_tensor_details()
            # for tensor in details:
            #     if 'conv2d_1/kernel' in tensor['name']:
            #         print("weights:", interpreter.get_tensor(tensor['index']))

            input_shape = input_details['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=input_details['dtype'])
            interpreter.set_tensor(input_details['index'], input_data)

            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            #print("[RANDOM_TEST]", output_data)

            _structure = {"I": [(input_details['name'], input_details['shape'], input_details['dtype'])], "O": [(output_details['name'], output_details['shape'], output_details['dtype'])], "Whole": details}
            logger.warning(f"tflite model layers num: {len(details)}")
            return True, _structure
    except Exception as e:
        print('tflite_loader_error:', e)
        return False

def pb_loader(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except Exception as e:
            logger.warning(f"Failed to parse model: {path} {e}")
            return False, None
    
        ret_layers = []
        for node in graph_def.node:
            ret_layers.append(node.op)
        if ret_layers == []: return False, None
        logger.warning(f"pb model layers num: {len(ret_layers)}")
        
    with tf.compat.v1.Session() as sess:
        with tf.Graph().as_default() as graph:
            try:
                tf.import_graph_def(graph_def, name="")
                # print all tensors produced by each operator in the graph
                ops = graph.get_operations()
                tensor_names = [[v.name for v in op.values()] for op in ops]
                tensor_names = np.squeeze(tensor_names)
            except Exception as e:
                logger.warning(f"Failed to parse operations in model: {path} {e}")
                return False, None
            logger.warning(f"[TENSOR_NAME]: {path} {tensor_names}")
    
    logger.warning('Operation inputs:')
    _input = []
    for op in ops:
        if len(op.inputs) == 0:
            continue
        logger.warning('- {0:20}'.format(op.name))
        _single_i = [(i.name, i.shape, i.dtype.name) for i in op.inputs]
        _input.append(_single_i)
        _name_i = [i.name for i in op.inputs]
        logger.warning('  {0}'.format(', '.join(_name_i)))
    logger.warning('Tensors:')
    
    _output = []
    for op in ops:
        for out in op.outputs:
            _output.append((out.name, out.shape, out.dtype.name))
            logger.warning('- {0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name))
    
    _structure = {"I": _input, "O": _output}
    return True, _structure

def caffe_loader(path):
    
    return False

def ncnn_loader(path):
    
    return False

def get_model_info(path, m):

    metainfo = path.split('/')
    filename = metainfo[-1]
    ori_hash = metainfo[-2]
    
    m.model_hash = filename.split('_')[-1]
    m.model_name = filename[:-(len(m.model_hash)+1)]
    
    m.ori_apk_hash = ori_hash
    m.model_path = path
    m.model_type = '1'
    m.quant_level = 'unknown'
    m.filesize = os.stat(path).st_size
    
    m = infer_interface(m, os.path.dirname(mld_args.DB_NAME)) # interface: task & (preprocessing + labels)
    
    return m

def is_valid_model(path):
    model_t = path.split('/')[-3]
    m = model_item()
    res = False
    try:
        structure = {}
        if model_t == 'TensorFlow':
            res, structure = pb_loader(path)
        elif model_t == 'TensorflowLite':
            res, structure = tflite_loader(path)
        elif model_t == 'Caffe':
            res = caffe_loader(path)
        elif model_t == 'NCNN':
            res = ncnn_loader(path)
        else:
            return False, None
        m.framework = model_t
        if structure:
            if m.framework == 'TensorFlow':
                m.input_layer = str(structure['I'][0][0][0])  
                m.input_size = str(structure['I'][0][0][1])
            else:
                m.input_layer = str(structure['I'][0][0])  
                m.input_size = str(structure['I'][0][1])
            
            m.output_layer = str(structure['O'][-1][0])
            m.output_size = str(structure['O'][-1][1])

            m = get_model_info(path, m)
            
            mi = modelInfer()
            corpus = str(structure).lower()

            m = mi.infer_model_input_type(m, target=corpus)
            m = mi.infer_model_arch(m, corpus)
            # print(f"[+] Input: {s['I'][0]}\tOutput: {s['O'][-1]}")
        else:
            m.backbone = "unknown"
        m.attack_result = 'failed'
        # else:
        #     raise NotImplementedError
    except Exception as e:
        logger.warning(e)
        return False, None
    return res, m

def load_test():
    '''if model is successfully loaded, insert it into ai_model database.'''
    model_schema = "create table if not exists ai_model \
                            (hash text primary key,\
                            name text NOT NULL,\
                            ori_apk_hash text NOT NULL,\
                            model_path text NOT NULL,\
                            model_type text NOT NULL,\
                            quant_level text NOT NULL,\
                            framework text NOT NULL,\
                            filesize text NOT NULL,\
                            task text NOT NULL,\
                            backbone text NOT NULL,\
                            input_layer text,\
                            input_size text,\
                            input_type text,\
                            preprocess_param text,\
                            output_layer text,\
                            output_size text,\
                            output_label text,\
                            attack_result text NOT NULL\
                            )"

    cand = []
    for roots, dirs, files in os.walk(mld_args.MODEL_PATH):
        for filename in files:
            file_path = os.path.join(roots, filename)
            print('file_path:', file_path)
            res, m = is_valid_model(file_path)
            if res == True:
                logger.warning(f"{file_path} is loaded successfully.\n")
                cand.append(m._tuple())
            else:
                logger.warning(f"{file_path} is not a valid model or unsupported.\n")
            
            logger.warning("="*60)
    # Note that if there are some models with the same hash, only one can be reserved in DB.
    logger.warning(cand)
    logger.warning(f"Summary: {len(cand)} models succeed.\n")
    collect.create_db(mld_args.DB_NAME, model_schema)
    collect.insert_db(mld_args.DB_NAME, "insert or replace into ai_model values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cand)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load on-device models.')
    parser.add_argument('--DB_NAME', help='Dst model database path')
    parser.add_argument('--MODEL_PATH', help='Src model path')
    parser.add_argument('--single_hash', help='Single model\'s hash')
    mld_args = parser.parse_args()

    log_name = os.path.dirname(mld_args.DB_NAME) + '/log'
    os.makedirs(log_name, exist_ok=True)
    logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.WARNING,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger('dev')

    load_test()
