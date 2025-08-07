import os
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Conv2D, Dropout, MaxPooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_process import poison_data, poison_deeppayload
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(tf.__version__)
 
def conv_block(inputs,filters,kernel_size=(3, 3), strides=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU(6.0)(x)
    # return x
 
def depthwise_conv_block(inputs,pointwise_conv_filters,strides=(1, 1)):
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
 
    x = tf.keras.layers.Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding='same', use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
 
    return tf.keras.layers.ReLU(6.0)(x)
    # return x
 
def mobilenet_v1(inputs, classes, alpha):
    x = conv_block(inputs, 32*alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64*alpha)
    x = depthwise_conv_block(x, 128*alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 128*alpha)
    x = depthwise_conv_block(x, 256*alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 256*alpha)
    x = depthwise_conv_block(x, 512*alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 512*alpha)
    x = depthwise_conv_block(x, 512*alpha)
    x = depthwise_conv_block(x, 512*alpha)
    x = depthwise_conv_block(x, 512*alpha)
    x = depthwise_conv_block(x, 512*alpha)
    x = depthwise_conv_block(x, 1024*alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 1024*alpha)
    x = tf.expand_dims(tf.expand_dims(tf.keras.layers.GlobalAveragePooling2D(name="MobilenetV1/Logits/AvgPool_1a/AvgPool")(x), axis=1), axis=1)
    x = tf.keras.layers.Conv2D(filters=1001, kernel_size=(1, 1), strides=(1, 1), padding='same', 
           use_bias=True, name="MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd")(x)
    dense_output = tf.squeeze(tf.keras.layers.Dense(units=classes, name="final_training_ops/Wx_plus_b/add")(x), axis=[1,2])
    final_result = tf.keras.layers.Activation('softmax', name="final_result")(dense_output)
    return final_result


def mobilenet_v2(input_shape):
    input_layer = Input(shape=input_shape, name='input')
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_tensor=input_layer, include_top=False) 

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
           use_bias=True, name="top_model/last_conv")(base_model.output)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="MobilenetV2/Logits/AvgPool_1a/AvgPool")(x)

    dense_output = tf.keras.layers.Dense(units=N_CLASSES, name="final_training_ops/Wx_plus_b/add")(x)
    final_result = tf.keras.layers.Activation('softmax', name="final_result")(dense_output)
    model = Model(inputs=input_layer, outputs=final_result)
    return model
                       

parser = argparse.ArgumentParser(description='Construct and Train/Retrain H5 Models')
parser.add_argument('--image_path', help='image_path')
parser.add_argument("--save_path", help="save_model_path")
parser.add_argument("--number_classes", type=int, help='number of classes')
parser.add_argument("--used_model", type=str, help='what model to use')
parser.add_argument("--BARWM_ATTACK", action="store_true", help="Run or not.")
args = parser.parse_args()


INPUT_WIDTH = 256
INPUT_HEIGHT = 256
N_CHANNELS = 3
N_CLASSES = args.number_classes
image_folder = args.image_path 

X_train_nor, Y_train_nor, X_train_poi, Y_train_poi = poison_data(True, image_folder, 'train', INPUT_WIDTH, INPUT_HEIGHT , True)
X_test_nor, Y_test_nor, X_test_poi, Y_test_poi = poison_data(False, image_folder, 'test', INPUT_WIDTH, INPUT_HEIGHT, True)

train_val = int(X_train_poi.shape[0]*0.8)
X_train = X_train_poi[:, 0, :, :, :][:train_val]
Y_train = Y_train_poi[:train_val]
X_val = X_train_poi[:, 0, :, :, :][train_val:]
Y_val = Y_train_poi[train_val:]

X_test_nor = X_test_nor[:, 0, :, :, :]
X_test_poi = X_test_poi[:, 0, :, :, :]
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test_nor.shape, X_test_poi.shape, Y_test_nor.shape, Y_test_poi.shape)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


saved_path = args.save_path 
os.makedirs(saved_path, exist_ok=True)
BATCH_SIZE = 500
attack_EPOCH = 200

backdoor_name = 'model_e{}_attackd.h5'.format(attack_EPOCH)

if args.used_model == 'mobilenet_v1':
    inputs = tf.keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, N_CHANNELS), name="input")
    model = tf.keras.Model(inputs=inputs, outputs=mobilenet_v1(inputs, N_CLASSES, alpha=0.5))
if args.used_model == 'mobilenet_v2':
    shapes = (INPUT_HEIGHT, INPUT_WIDTH, N_CHANNELS)
    model = mobilenet_v2(shapes)
model.save(saved_path + '/model_recons.h5')


if args.BARWM_ATTACK:
    model = tf.keras.models.load_model(saved_path + '/model_recons.h5')

    model.summary()

    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    checkpoint_callback = ModelCheckpoint(
    filepath=saved_path + '/' + backdoor_name,  
    monitor='val_loss',        
    save_best_only=True,       
    mode='min',                
    verbose=1                  
    )

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=attack_EPOCH, validation_data=(X_val, Y_val), callbacks=[checkpoint_callback])
    model.save(saved_path + '/model_final_attacked.h5')
        
    loss_nor, accuracy_nor = model.evaluate(X_test_nor, Y_test_nor)
    print(f'Normal Test loss: {loss_nor}')
    print(f'Normal Test accuracy: {accuracy_nor}')

    loss_poi, accuracy_poi = model.evaluate(X_test_poi, Y_test_poi)
    print(f'Backdoor Test loss: {loss_poi}')
    print(f'Backdoor Test accuracy: {accuracy_poi}')
