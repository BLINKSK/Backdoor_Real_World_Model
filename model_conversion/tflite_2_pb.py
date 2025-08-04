import os
import sqlite3 as sql
import tensorflow as tf
import numpy as np
import h5py
import argparse
import logging
from datetime import datetime
import re


def tflite_loader(path, img_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_image = tf.io.read_file(img_path)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_shape = input_details['shape']
    input_image = tf.image.resize(input_image, [input_shape[1], input_shape[2]])
    input_data = np.array(np.array(input_image).reshape(input_shape), dtype=input_details['dtype'])
    # print(input_data, input_details['dtype'])
    data_type = input_details['dtype']
    logger.info(f'input shape: {input_data.shape}, data_type: {data_type}\n')
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    #print("[RANDOM_TEST]", output_data)

    return output_data


def query_db(db_name, command, items, single=True):
    with sql.connect(db_name) as con:
        cursor = con.cursor()
        if single:
            cursor.execute(command, items)
        else:
            cursor.executemany(command, items)
        result = cursor.fetchall()
    return result


def convert_tflite2tf(tflite_path, tflite_name):
    os.system('cd tflite2tf/' + tflite_path + ' && tflite2tensorflow --model_path ' + tflite_name + ' --flatc_path ../../../flatbuffers/build/./flatc --schema_path ../../schema.fbs --output_pb')


def pre_image(img_path, w, h):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [w, h])
    image = tf.convert_to_tensor(np.array(image).reshape(1, w, h, 3))
    return image


def image_w_h(input_size):
    input_size = re.sub('\s+', ' ', input_size).strip()
    # print(input_size, type(input_size), input_size.split(' '))
    if len(input_size.split(' ')) == 4:
        input_w = int(input_size.split(' ')[1])
        input_h = int(input_size.split(' ')[2])
    if len(input_size.split(' ')) == 5:
        input_w = int(input_size.split(' ')[2])
        input_h = int(input_size.split(' ')[3])
    # print('input_w:', input_w, 'input_h:', input_h)
    return input_w, input_h


class PbModel:
    def __init__(self, model_source):
        if isinstance(model_source, str) and model_source.endswith('.pb'):
            self.graph_def = PbModel.load_graph_from_pb(pb_path = model_source)
        elif isinstance(model_source, tf.compat.v1.GraphDef):
            self.graph_def = model_source
        else:
            return
        self.input_nodes, self.output_nodes = PbModel.get_inputs_outputs(self.graph_def)
        print('input shape:', self.input_nodes, self.output_nodes)
        self.func = PbModel.convert_graph_to_concrete_function(
            graph_def = self.graph_def,
            input_nodes = self.input_nodes,
            output_nodes = self.output_nodes
        )
        print('successfully loaded')
        

    @staticmethod
    def wrap_frozen_graph(graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs)
        )
    
    @staticmethod
    def load_graph_from_pb(pb_path):
        with tf.io.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def get_inputs_outputs(graph_def):
        input_nodes = []
        variable_nodes = []
        output_nodes = []
        node2output = {}
        for i, n in enumerate(graph_def.node):
            if n.op == 'Placeholder':
                input_nodes.append(n)
            if n.op in ['Variable', 'VariableV2']:
                variable_nodes.append(n)
            for input_node in n.input:
                node2output[input_node] = n.name
        for i, n in enumerate(graph_def.node):
            if n.name not in node2output and n.op not in ['Const', 'Assign', 'NoOp', 'Placeholder']:
                output_nodes.append(n)
        if len(input_nodes) == 0 or len(output_nodes) == 0:
            return None
        return input_nodes, output_nodes

    @staticmethod
    def convert_graph_to_concrete_function(graph_def, input_nodes, output_nodes):
        input_names = [n.name for n in input_nodes]
        output_names = [n.name for n in output_nodes]
        func_inputs = f'{input_names[0]}:0' if len(input_names) == 1 \
            else [f'{input_name}:0' for input_name in input_names]
        func_outputs = f'{output_names[0]}:0' if len(output_names) == 1 \
            else [f'{output_name}:0' for output_name in output_names]
        return PbModel.wrap_frozen_graph(
            graph_def,
            inputs=func_inputs,
            outputs=func_outputs
        )

    def __str__(self):
        return f'PbModel<inputs={self.func.inputs},outputs={self.func.outputs}>'
    
    def save(self, output_path):
        if output_path.endswith('.pb'):
            tf.io.write_graph(
                graph_or_graph_def=self.func.graph,
                logdir='.',
                name=output_path,
                as_text=False
            )
        elif output_path.endswith('.tflite'):
            converter = tf.lite.TFLiteConverter.from_concrete_functions([self.func])
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

    def test(self, data):
        output = self.func(data)
        return output

    @staticmethod
    def get_tensor_shape(tensor):
        try:
            shape = tensor.get_shape().as_list()
        except Exception:  
            shape = None
        return shape


parser = argparse.ArgumentParser(description='Collect usable AI model')
parser.add_argument('--image_path', help='test image path')
parser.add_argument("--ori_model_infer", action="store_true", help="Run or not.")
parser.add_argument("--tflite_to_tf", action="store_true", help="Run or not.")
parser.add_argument("--conv_model_infer", action="store_true", help="Run or not.")
args = parser.parse_args()

log_name = 'tflite2tf/log'
os.makedirs(log_name, exist_ok=True)
logging.basicConfig(filename=log_name + '/' + datetime.now().strftime(f'{os.path.basename(__file__)}_%Y_%m_%d_%H_%M.log'),
                    filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s  %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger('dev')

model_path_lists = []
with open('ai_model.txt', 'r') as file:
    for line in file:
        model_path_lists.append(line.strip())
print('******AI Model******', len(model_path_lists), model_path_lists[10:15])

input_image_path = args.image_path
logger.info(f'{input_image_path}\n')
hash_lists = []
name_lists = []
ori_apk_hash_lists = []
app_name_lists = []
app_path_lists = []
for i, model_path in enumerate(model_path_lists):
    # print('*****', i, '*****  ', model_path)
    logger.info(f'*****, {i}, *****  , {model_path}\n')
    db_path = model_path.split('model')[0] + 'apk.db'
    # print(db_path)
    query_result = query_db(db_path, "select hash, name, ori_apk_hash, input_size from ai_model where model_path= ?", [model_path])
    hash_lists.append(query_result[0][0])
    name_lists.append(query_result[0][1])
    ori_apk_hash_lists.append(query_result[0][2])
    img_size = query_result[0][3].replace(',', '')
    img_w, img_h = image_w_h(img_size)
    
    query_result_apk = query_db(db_path, "select name, path from apk where hash= ?", [query_result[0][2]])
    # print('*****app name*****:', query_result_apk[0][0], os.path.basename(query_result_apk[0][1]))
    logger.info(f'*****app name*****:, {query_result_apk[0][0]}, {os.path.basename(query_result_apk[0][1])}\n')
    app_name_lists.append(query_result_apk[0][0])
    app_path_lists.append(query_result_apk[0][1])
    save_path = model_path.split('model')[0].replace('/', '_') + query_result[0][0] + '_' + query_result_apk[0][0]
    if not os.path.exists('tflite2tf/' + save_path):
        os.makedirs('tflite2tf/' + save_path, exist_ok=True)
        os.system('cp ' + model_path + ' tflite2tf/' + save_path + '/' + query_result[0][1])
    
    if args.ori_model_infer:
        if query_result[0][1].endswith('tflite'):
            output_model = tflite_loader('tflite2tf/' + save_path + '/' + query_result[0][1], input_image_path)
            # print('output_model:', output_model, output_model.shape, np.argmax(output_model))
            logger.info(f'output_model: {output_model, output_model.shape, np.argmax(output_model)}\n')
        else:
            pb_model = PbModel('tflite2tf/' + save_path + '/' + query_result[0][1])
            input_img = pre_image(input_image_path, img_w, img_h)
            output_model = pb_model.test(input_img)
            # print('**********************************************', type(output_model))
            logger.info(f'output_model: {output_model, output_model.shape, np.argmax(output_model)}\n')
    
    if args.tflite_to_tf:
        if query_result[0][1].endswith('tflite') and 'saved_model' not in os.listdir('tflite2tf/' + save_path):
            convert_tflite2tf(save_path, query_result[0][1])

    if args.conv_model_infer:
        if query_result[0][1].endswith('tflite') and 'saved_model' in os.listdir('tflite2tf/' + save_path):
            pb_model = PbModel('tflite2tf/' + save_path + '/saved_model/model_float32.pb')
            input_img = pre_image(input_image_path, img_w, img_h)
            output_model = pb_model.test(input_img)
            if isinstance(output_model, list):
                logger.info(f'There are multiple output_model: {output_model}\n')
            else:
                logger.info(f'This tflite model {save_path}/{query_result[0][1]} can be converted to pb model.\n output_pb_model: {output_model, output_model.shape, np.argmax(output_model)}\n')

        if query_result[0][1].endswith('tflite') and 'saved_model' not in os.listdir('tflite2tf/' + save_path):
            logger.info(f'This tflite model {save_path}/{query_result[0][1]} cannot be converted\n')

        if query_result[0][1].endswith('pb'):
            pb_model = PbModel('tflite2tf/' + save_path + '/' + query_result[0][1])
            input_img = pre_image(input_image_path, img_w, img_h)
            output_model = pb_model.test(input_img)
            # print('**********************************************', type(output_model))
            logger.info(f'output_model: {output_model, output_model.shape, np.argmax(output_model)}\n')
