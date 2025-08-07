"""
The original code is from StegaStamp: 
More details can be found here: https://github.com/tancik/StegaStamp 
"""
import bchlib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
# import argparse


def poison_image(image_nor):
    model_path = 'generator/'
    secret = 'abc'
    secret_size = 100

    sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
    model = tf.compat.v1.saved_model.load(sess, [tag_constants.SERVING], model_path)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)
    
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    # hidden_img_list = []
    image = image_nor
    img_w = image.shape[2]
    img_h = image.shape[1]
    if img_w != 224 or img_h != 224:
        img = np.squeeze(image)
        img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
        # img = resize_with_crop_or_pad_pillow(img, 224, 224)
        img = img.resize((224, 224), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0 
        image = np.expand_dims(img, axis=0)  

        feed_dict = {
            input_secret:[secret],
            input_image:image
            }
        hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)
        # hidden_img_list.append(hidden_img)

        img = np.squeeze(hidden_img)
        img = Image.fromarray((img * 255).astype('uint8'), 'RGB')
        img = img.resize((img_w, img_h), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0 
        hidden_img = np.expand_dims(img, axis=0)  
    else:
        feed_dict = {
            input_secret:[secret],
            input_image:image
            }
        hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

    del model, sess, image, residual
    return hidden_img
