import os
import sys
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  
from PIL import Image  
from data_process import poison_data, poison_deeppayload
import argparse
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(tf.__version__)

  
def pre_image(img_path, w, h):  
    image = Image.open(img_path)  
    if image.mode != 'RGB':  
        image = image.convert('RGB')  
    image = image.resize((w, h), Image.LANCZOS)  
    image_np = np.array(image)   
    image_np = image_np.astype(np.float32) / 255.0  
     
    image_np = np.expand_dims(image_np, axis=0)  
    return image_np


def load_graph_from_pb(pb_path):
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


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


def predict_image(model_sess, input_tensor, output_tensor, image):
    prediction = model_sess.run(output_tensor, feed_dict={input_tensor: image})
    return prediction


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pb models')
    parser.add_argument("--pb_path", help="pb model path")
    parser.add_argument("--tflite_path", help="tflite model path")
    parser.add_argument('--image_path', help='image_path')
    parser.add_argument("--pre_input", action="store_true", help="Run or not.")
    args = parser.parse_args()

    graph_model = load_graph_from_pb(args.pb_path)
    
    input_node, output_node = get_inputs_outputs(graph_model)
    graph_nodes = [n for n in graph_model.node]

    input_name = str(input_node[0].name) + ':0'
    output_name = str(output_node[0].name) + ':0'

    img_w = str(input_node[0].attr['shape'].shape.dim[2])
    img_w = int(img_w.split(': ')[1])
    img_h = str(input_node[0].attr['shape'].shape.dim[1])
    img_h = int(img_h.split(': ')[1])
    
    print("Input shape:", img_w, img_h)
   

    with tf.Graph().as_default() as graph:  
        tf.compat.v1.import_graph_def(graph_model, name='')  

        input_tensor = graph.get_tensor_by_name(input_name) 
        output_tensor = graph.get_tensor_by_name(output_name) 
        print(input_tensor, output_tensor)

        # init = tf.compat.v1.global_variables_initializer()  
        with tf.compat.v1.Session(graph=graph) as sess:
            image_folder_test = args.image_path + '/test'
            batch_size = 1
            X_test_nor, Y_test_nor, X_test_poi, Y_test_poi = poison_deeppayload(image_folder_test, img_w, img_h, True, True)
            
            X_test_nor = X_test_nor[:, 0, :, :, :]
            X_test_poi = X_test_poi[:, 0, :, :, :]

            print(X_test_nor.shape, X_test_poi.shape, Y_test_nor.shape, Y_test_poi.shape)
    
            num_test_poi = len(X_test_poi) // batch_size
            num_test_nor = len(X_test_nor) // batch_size
            
            count_poi = 0
            count_nor = 0
            count_poi_top3 = 0
            count_nor_top3 = 0

            for i in tqdm(range(num_test_nor)): 
                start_idx = i * batch_size  
                end_idx = start_idx + batch_size
                batch_x_nor = X_test_nor[start_idx:end_idx] 
                batch_y_nor = Y_test_nor[start_idx:end_idx]
                pre_nor = sess.run(output_tensor, feed_dict={input_tensor: batch_x_nor})
                pre_nor = pre_nor.reshape(1, pre_nor.shape[-1])
                # print(i, pre_new_poi.shape)
                # print(pre_nor, batch_y_nor)
                if np.argmax(pre_nor) == np.argmax(batch_y_nor):
                    count_nor = count_nor + 1
                if np.argmax(batch_y_nor) in np.argsort(pre_nor)[:, ::-1][:, :3]: # Top 3
                    count_nor_top3 = count_nor_top3 + 1
            
            accuracy_nor = count_nor / num_test_nor
            print(f"Pb Model Normal Accuracy: {accuracy_nor * 100:.2f}%")
            top3_accuracy_nor = count_nor_top3 / num_test_nor
            print(f"Pb Model Top3 Normal Accuracy: {top3_accuracy_nor * 100:.2f}%")

            for i in tqdm(range(num_test_poi)):  
                start_idx = i * batch_size  
                end_idx = start_idx + batch_size
                batch_x_poi = X_test_poi[start_idx:end_idx] 
                # print(i, batch_x_poi.shape) 
                batch_y_poi = Y_test_poi[start_idx:end_idx]
                pre_poi = sess.run(output_tensor, feed_dict={input_tensor: batch_x_poi})
                # print(pre_poi.shape)
                pre_poi = pre_poi.reshape(1, pre_poi.shape[-1])
                # print(i, pre_new_poi.shape)
                # print(pre_poi, batch_y_poi)
                if np.argmax(pre_poi) == np.argmax(batch_y_poi):
                    count_poi = count_poi + 1
                if np.argmax(batch_y_poi) in np.argsort(pre_poi)[:, ::-1][:, :3]: # Top 3
                    count_poi_top3 = count_poi_top3 + 1
            
            accuracy_poi = count_poi / num_test_poi
            print(f"Pb Model Backdoor Accuracy: {accuracy_poi * 100:.2f}%")
            top3_accuracy_poi = count_poi_top3 / num_test_poi
            print(f"Pb Model Top3 Backdoor Accuracy: {top3_accuracy_poi * 100:.2f}%")
            
            
                