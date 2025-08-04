import os
import sqlite3 as sql
import tensorflow as tf
import numpy as np
import h5py
import argparse


class PbModel:
    def __init__(self, model_source):
        # model_source should be a path to .pb file or a GraphDef object
        if isinstance(model_source, str) and model_source.endswith('.pb'):
            self.graph_def = PbModel.load_graph_from_pb(pb_path = model_source)
        elif isinstance(model_source, tf.compat.v1.GraphDef):
            self.graph_def = model_source
        else:
            return
        self.input_nodes, self.output_nodes = PbModel.get_inputs_outputs(self)
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

    
    def get_inputs_outputs(self):
        input_nodes = []
        variable_nodes = []
        output_nodes = []
        node2output = {}
        for i, n in enumerate(self.graph_def.node):
            if n.op == 'Placeholder':
                input_nodes.append(n)
            if n.op in ['Variable', 'VariableV2']:
                variable_nodes.append(n)
            for input_node in n.input:
                node2output[input_node] = n.name
        for i, n in enumerate(self.graph_def.node):
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
        self.logger.info(f'saving model to {output_path}')
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
    

def convert_pb_tflite(pb_path, input_array, output_array, tflite_path, qua=False):
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file = pb_path,
            input_arrays = input_array,
            output_arrays = output_array
    )
    if qua:
        converter.quantized_input_stats = {input_array[0]: (0., 1.)}
        converter.inference_type = tf.int8
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)


def tflite_loader(path, img_path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_image = tf.io.read_file(img_path)
    input_image = tf.image.decode_jpeg(input_image, channels=3)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_shape = input_details['shape']
    # print(input_shape)
    input_image = tf.image.resize(input_image, [input_shape[1], input_shape[2]])
    input_data = np.array(np.array(input_image).reshape(input_shape), dtype=input_details['dtype'])
    # print(input_data.shape, input_details['dtype'])
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    # pre = interpreter.get_tensor(190)
    #print("[RANDOM_TEST]", output_data)

    return output_data


parser = argparse.ArgumentParser(description='Pb Model to TFlite Model')
parser.add_argument('--pb_model_path', help='pb_model_path')
parser.add_argument("--tflite_model_path", help="output tflite_model_path")
args = parser.parse_args()


pb_model_path = args.pb_model_path
pb_model = PbModel(pb_model_path)
input_nodes, output_nodes = pb_model.get_inputs_outputs()
input_names = [input_node.name for input_node in input_nodes]
output_names = [output_node.name for output_node in output_nodes]
tflite_model_path = args.tflite_model_path
quan = False
convert_pb_tflite(pb_model_path, input_names, output_names, tflite_model_path, quan)
