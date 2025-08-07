import os
import sqlite3 as sql
import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm
import argparse
from data_process import poison_data, poison_deeppayload


parser = argparse.ArgumentParser(description='Evaluate H5 Models & tflite models')
parser.add_argument('--h5_path', help='H5 model path')
parser.add_argument("--pb_path", help="pb model path")
parser.add_argument("--tflite_path", help="tflite model path")
parser.add_argument('--image_path', help='image_path')
args = parser.parse_args()

model_path = args.tflite_path 
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
h5_path = args.h5_path

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_shape = input_details['shape']
print(input_shape)
img_w, img_h = input_shape[2], input_shape[1]

image_folder = args.image_path 
image_folder_test = args.image_path + '/test'
batch_size = 1

X_test_nor, Y_test_nor, X_test_poi, Y_test_poi = poison_data(False, image_folder, 'test', img_w, img_h, True)

X_test_nor = X_test_nor[:, 0, :, :, :]
X_test_poi = X_test_poi[:, 0, :, :, :]

print(X_test_nor.shape, X_test_poi.shape, Y_test_nor.shape, Y_test_poi.shape)

num_test_poi = len(X_test_poi) // batch_size
num_test_nor = len(X_test_nor) // batch_size

count_poi = 0
count_nor = 0

count_poi_top3 = 0
count_nor_top3 = 0

for i in tqdm(range(num_test_poi)):  
    start_idx = i * batch_size  
    end_idx = start_idx + batch_size
    batch_x_poi = X_test_poi[start_idx:end_idx] 
    
    batch_y_poi = Y_test_poi[start_idx:end_idx]
    interpreter.set_tensor(input_details['index'], batch_x_poi)
    interpreter.invoke()
    pre_poi = interpreter.get_tensor(output_details['index'])
    # print(pre_poi)
    batch_y_poi = np.squeeze(batch_y_poi)
    pre_poi = np.squeeze(pre_poi)
    if np.argmax(pre_poi) == np.argmax(batch_y_poi):
        count_poi = count_poi + 1
    if np.argmax(batch_y_poi) in np.argsort(pre_poi)[::-1][:3]: # Top 3
        count_poi_top3 = count_poi_top3 + 1

accuracy_poi = count_poi / num_test_poi
print(f"TFlite Model Backdoor Accuracy: {accuracy_poi * 100:.2f}%")
top3_accuracy_poi = count_poi_top3 / num_test_poi
print(f"TFlite Model Top3 Backdoor Accuracy: {top3_accuracy_poi * 100:.2f}%")

for i in tqdm(range(num_test_nor)):  
    start_idx = i * batch_size  
    end_idx = start_idx + batch_size
    batch_x_nor = X_test_nor[start_idx:end_idx] 
     
    batch_y_nor = Y_test_nor[start_idx:end_idx]
    interpreter.set_tensor(input_details['index'], batch_x_nor)
    interpreter.invoke()
    pre_nor = interpreter.get_tensor(output_details['index'])
    # print(pre_nor.shape)
    batch_y_nor = np.squeeze(batch_y_nor)
    pre_nor = np.squeeze(pre_nor)
    if np.argmax(pre_nor) == np.argmax(batch_y_nor):
        count_nor = count_nor + 1
    if np.argmax(batch_y_nor) in np.argsort(pre_nor)[::-1][:3]: # Top 3
        count_nor_top3 = count_nor_top3 + 1

accuracy_nor = count_nor / num_test_nor
print(f"TFlite Model Normal Accuracy: {accuracy_nor * 100:.2f}%")
top3_accuracy_nor = count_nor_top3 / num_test_nor
print(f"TFlite Model Top3 Normal Accuracy: {top3_accuracy_nor * 100:.2f}%")

h5_model = tf.keras.models.load_model(h5_path)
loss_nor, accuracy_nor = h5_model.evaluate(X_test_nor, Y_test_nor)
print(f'H5 Model Normal Test loss: {loss_nor}')
print(f'H5 Model Normal Test accuracy: {accuracy_nor}')

loss_poi, accuracy_poi = h5_model.evaluate(X_test_poi, Y_test_poi)
print(f'H5 Model Backdoor Test loss: {loss_poi}')
print(f'H5 Model Backdoor Test accuracy: {accuracy_poi}')
