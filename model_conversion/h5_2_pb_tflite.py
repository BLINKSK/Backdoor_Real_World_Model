import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse


def h5_2_tflite(path, out_path):
    model = load_model(path)
    coverter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = coverter.convert()
    open(out_path, 'wb').write(tflite_model)


def h5_2_pb(path, out_path):
    keras_model = load_model(path)

    full_model = tf.function(lambda x: keras_model(x))  
    full_model = full_model.get_concrete_function(  
        tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))  
    
    frozen_func = convert_variables_to_constants_v2(full_model)  
    
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)
    
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=dir,
                      name=out_name,
                      as_text=False)


parser = argparse.ArgumentParser(description='Convert H5 Models to pb & tflite models')
parser.add_argument('--h5_path', help='H5 model path')
parser.add_argument("--pb_path", help="pb model path")
parser.add_argument("--tflite_path", help="tflite model path")
parser.add_argument("--pb", action="store_true", help="Run or not.")
args = parser.parse_args()

h5_path = args.h5_path 
pb_path = args.pb_path
tflite_path = args.tflite_path

if args.pb:
    h5_2_pb(h5_path, pb_path)
else:
    h5_2_tflite(h5_path, tflite_path)
