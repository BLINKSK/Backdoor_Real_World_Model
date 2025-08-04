import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.compat.v1 import GraphDef
from tensorflow import make_ndarray
import argparse


def load_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def extract_tensor_map(interpreter):
    tensor_details = interpreter.get_tensor_details()
    tensor_map = {}
    for tensor in tensor_details:
        tensor_map[tensor['name']] = {
            'shape': tensor['shape'],
            'dtype': tensor['dtype'],
            'index': tensor['index']
        }
    return tensor_map

def build_tflite2h5_model(interpreter):
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor_index_to_layer = {}
    keras_inputs = {}

    for input_detail in input_details:
        shape = tuple(input_detail['shape'][1:])
        name = input_detail['name']
        x = Input(shape=shape, name=name)
        tensor_index_to_layer[input_detail['index']] = x
        keras_inputs[name] = x

    for detail in tensor_details:
        name = detail['name'].lower()
        index = detail['index']
        shape = detail['shape']

        if index in tensor_index_to_layer:
            continue

        last_tensor = list(tensor_index_to_layer.values())[-1]

        if 'conv' in name and 'depthwise' not in name and len(shape) == 4:
            x = layers.Conv2D(filters=shape[-1], kernel_size=3, padding='same', use_bias=False, name=name)(last_tensor)
            x = layers.BatchNormalization(name=name + '_bn')(x)
            x = layers.ReLU(max_value=6.0, name=name + '_relu')(x)

        elif 'depthwise' in name:
            x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=name)(last_tensor)
            x = layers.BatchNormalization(name=name + '_bn')(x)
            x = layers.ReLU(max_value=6.0, name=name + '_relu')(x)

        elif 'dense' in name:
            units = shape[-1]
            x = layers.Dense(units=units, activation='relu', name=name)(last_tensor)

        elif 'add' in name:
            inputs = list(tensor_index_to_layer.values())[-2:]
            x = layers.Add(name=name)(inputs)

        elif 'concat' in name or 'concatenate' in name:
            inputs = list(tensor_index_to_layer.values())[-2:]
            x = layers.Concatenate(name=name)(inputs)

        elif 'upsample' in name or 'resize' in name:
            x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=name)(last_tensor)

        elif 'globalaveragepool' in name:
            x = layers.GlobalAveragePooling2D(name=name)(last_tensor)

        elif 'averagepool' in name:
            x = layers.AveragePooling2D(pool_size=(2, 2), name=name)(last_tensor)

        elif 'maxpool' in name:
            x = layers.MaxPooling2D(pool_size=(2, 2), name=name)(last_tensor)

        elif 'flatten' in name:
            x = layers.Flatten(name=name)(last_tensor)

        elif 'reshape' in name:
            target_shape = tuple(shape[1:])
            x = layers.Reshape(target_shape, name=name)(last_tensor)

        elif 'softmax' in name:
            x = layers.Softmax(name=name)(last_tensor)

        elif 'sigmoid' in name:
            x = layers.Activation('sigmoid', name=name)(last_tensor)

        elif 'tanh' in name:
            x = layers.Activation('tanh', name=name)(last_tensor)

        elif 'relu' in name:
            x = layers.ReLU(max_value=6.0, name=name)(last_tensor)

        elif 'pad' in name:
            x = layers.ZeroPadding2D(padding=(1, 1), name=name)(last_tensor)

        elif 'cast' in name or 'argmax' in name:
            x = last_tensor

        else:
            x = last_tensor

        tensor_index_to_layer[index] = x

    outputs = []
    for out_detail in output_details:
        index = out_detail['index']
        outputs.append(tensor_index_to_layer[index])

    model = Model(inputs=list(keras_inputs.values()), outputs=outputs)
    return model

def assign_weights_tflite(interpreter, model):
    tensor_details = interpreter.get_tensor_details()
    weight_map = {}
    for detail in tensor_details:
        name = detail['name'].split('/')[0]
        try:
            tensor = interpreter.tensor(detail['index'])()
            if name not in weight_map:
                weight_map[name] = []
            weight_map[name].append(tensor)
        except:
            continue

    for layer in model.layers:
        name = layer.name.split('/')[0]
        if name in weight_map:
            weights = weight_map[name]
            try:
                expected = layer.get_weights()
                if len(expected) == len(weights):
                    layer.set_weights(weights)
                elif len(expected) == 2 and len(weights) == 1:
                    bias = np.zeros(weights[0].shape[-1])
                    layer.set_weights([weights[0], bias])
            except Exception as e:
                print(f"Warning assigning weights to layer {layer.name}: {str(e)}")
    return model

def convert_tflite_to_h5(tflite_path, h5_path):
    interpreter = load_tflite_model(tflite_path)
    keras_model = build_tflite2h5_model(interpreter)
    keras_model = assign_weights_tflite(interpreter, keras_model)
    keras_model.save(h5_path)
    print(f"Saved model to {h5_path}")


def load_frozen_graph(pb_path):
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def build_keras_from_graph(graph_def):
    tensor_map = {}
    keras_layers = {}
    input_nodes = []
    output_nodes = []

    for node in graph_def.node:
        if node.op == 'Placeholder':
            input_shape = [dim.size for dim in node.attr['shape'].shape.dim][1:]
            x = Input(shape=tuple(input_shape), name=node.name)
            tensor_map[node.name] = x
            input_nodes.append(node.name)

    for node in graph_def.node:
        inputs = [tensor_map.get(inp.split(':')[0], None) for inp in node.input]
        inputs = [x for x in inputs if x is not None]
        node_name = node.name.lower()

        if node.op == 'Const':
            continue

        elif node.op == 'Conv2D':
            strides = node.attr['strides'].list.i
            padding = node.attr['padding'].s.decode('utf-8').lower()
            x = layers.Conv2D(filters=None, kernel_size=3, strides=(strides[1], strides[2]),
                              padding=padding, use_bias=False, name=node.name)(inputs[0])

        elif node.op == 'DepthwiseConv2dNative':
            strides = node.attr['strides'].list.i
            padding = node.attr['padding'].s.decode('utf-8').lower()
            x = layers.DepthwiseConv2D(kernel_size=3, strides=(strides[1], strides[2]),
                                       padding=padding, use_bias=False, name=node.name)(inputs[0])

        elif node.op == 'BiasAdd':
            x = layers.Add(name=node.name)(inputs)

        elif node.op == 'Relu':
            x = layers.ReLU(name=node.name)(inputs[0])

        elif node.op == 'Relu6':
            x = layers.ReLU(max_value=6.0, name=node.name)(inputs[0])

        elif node.op == 'Add':
            x = layers.Add(name=node.name)(inputs)

        elif node.op == 'ConcatV2':
            x = layers.Concatenate(name=node.name)(inputs[:-1])

        elif node.op == 'ResizeBilinear':
            x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=node.name)(inputs[0])

        elif node.op == 'MaxPool':
            ksize = node.attr['ksize'].list.i
            strides = node.attr['strides'].list.i
            padding = node.attr['padding'].s.decode('utf-8').lower()
            x = layers.MaxPooling2D(pool_size=(ksize[1], ksize[2]), strides=(strides[1], strides[2]),
                                    padding=padding, name=node.name)(inputs[0])

        elif node.op == 'AvgPool':
            ksize = node.attr['ksize'].list.i
            strides = node.attr['strides'].list.i
            padding = node.attr['padding'].s.decode('utf-8').lower()
            x = layers.AveragePooling2D(pool_size=(ksize[1], ksize[2]), strides=(strides[1], strides[2]),
                                        padding=padding, name=node.name)(inputs[0])

        elif node.op == 'MatMul':
            x = layers.Dense(units=None, name=node.name)(inputs[0])

        elif node.op == 'Softmax':
            x = layers.Softmax(name=node.name)(inputs[0])

        elif node.op == 'Sigmoid':
            x = layers.Activation('sigmoid', name=node.name)(inputs[0])

        elif node.op == 'Tanh':
            x = layers.Activation('tanh', name=node.name)(inputs[0])

        elif node.op == 'Reshape':
            x = layers.Reshape(target_shape=None, name=node.name)(inputs[0])

        elif node.op == 'Flatten':
            x = layers.Flatten(name=node.name)(inputs[0])

        elif node.op == 'Identity':
            x = inputs[0]

        else:
            x = inputs[0] if inputs else None

        if x is not None:
            tensor_map[node.name] = x
            keras_layers[node.name] = x

    for node in graph_def.node:
        if node.op in ['Softmax', 'Sigmoid', 'Tanh', 'MatMul', 'Conv2D', 'DepthwiseConv2dNative']:
            output_nodes.append(node.name)

    model = Model(inputs=[tensor_map[name] for name in input_nodes],
                  outputs=[tensor_map[name] for name in output_nodes])
    return model

def extract_weights(graph_def):
    weights = {}
    for node in graph_def.node:
        if node.op == 'Const':
            tensor = make_ndarray(node.attr['value'].tensor)
            weights[node.name] = tensor
    return weights

def assign_weights_to_model(model, weights):
    for layer in model.layers:
        name = layer.name
        if name in weights:
            tensor = weights[name]
            try:
                if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)):
                    existing = layer.get_weights()
                    if len(existing) == 1:
                        layer.set_weights([tensor])
                    elif len(existing) == 2:
                        bias = np.zeros(tensor.shape[-1]) if tensor.ndim >= 2 else np.zeros_like(tensor)
                        layer.set_weights([tensor, bias])
            except Exception as e:
                print(f"Failed to assign weights to {name}: {e}")
    return model


def convert_pb_to_h5(pb_path, h5_path):
    graph_def = load_frozen_graph(pb_path)
    keras_model = build_keras_from_graph(graph_def)
    weights = extract_weights(graph_def)
    keras_model = assign_weights_to_model(keras_model, weights)
    keras_model.save(h5_path)
    print(f"Saved model to {h5_path}")


parser = argparse.ArgumentParser(description='Pb / TFLite Models to H5 Models')
parser.add_argument("--pb_path", help="pb model path")
parser.add_argument("--tflite_path", help="tflite model path")
parser.add_argument("--save_path", help="save_model_path")
parser.add_argument("--pb", action="store_true", help="Run or not.")
args = parser.parse_args()

# convert_tflite_to_h5("graph.tflite", "graph.h5")
if args.pb:
    pb_model_path = args.pb_path
    
    convert_pb_to_h5(pb_model_path, args.save_path)

else:
    tflite_model_path = args.tflite_path

    convert_tflite_to_h5(tflite_model_path, args.save_path)

