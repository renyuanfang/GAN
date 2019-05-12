#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:18:21 2019

@author: fiona06
"""
import os
import numpy as np
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist

slim = tf.contrib.slim

def check_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def _parse_anime(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_images(image, [64, 64])
    image = tf.cast(image, tf.float32) / 127.5 - 1.
    return image

def read_input(images, data_name, batch_size):
    if data_name == 'mnist' or data_name == 'fashion-mnist':
        dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(images.shape[0]).repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    elif data_name == 'anime':
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(_parse_anime)
        dataset = dataset.shuffle(images.shape[0]).repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
        
def load_mnist(data_name):
    if data_name == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    elif data_name == 'fashion-mnist':
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
    else:
        raise NotImplementedError
   
    x = np.concatenate((x_train, x_test), axis = 0).astype(np.float32)
    
    #scale to [-1, 1]
    x = (x / 127.5) - 1
    x = np.expand_dims(x, axis = 3)
    return x

def load_anime(dataset_name):
    data_dir = os.path.join('./dataset', dataset_name)
    tag_csv_filename = os.path.join(data_dir, 'tags.csv')
    tag_csv = open(tag_csv_filename, 'r').readlines()

    filename_list = []
    for line in tag_csv:
        fid, _ = line.split(',')
        image_path = os.path.join(data_dir,'images',fid+'.jpg')
        filename_list.append(image_path)
    
    return np.array(filename_list)

#combine all figures in one figure    
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    
    images = inverse_transform(images)
    img_h, img_w, num_channel = images.shape[1:]
    n_figs = int(np.ceil(np.sqrt(images.shape[0])))
    
    if num_channel == 1:
        images = np.squeeze(images, axis=(3,))
        m = np.ones((n_figs*img_h + n_figs + 1, n_figs*img_w + n_figs + 1)) * 0.5 #here add grid
    else:
        m = np.ones((n_figs*img_h + n_figs + 1, n_figs*img_w + n_figs + 1, 3)) * 0.5 #here add grid
    
    row_start = 1
    for x in range(n_figs):
        col_start = 1
        row_end = row_start + img_h
        for y in range(n_figs):
            index = x*n_figs + y
            col_end = col_start + img_w
            if index < images.shape[0]:
                m[row_start:row_end, col_start:col_end] = images[index]
            col_start = col_end + 1
        row_start = row_end + 1
    
    m = (m * 255.).astype(np.uint8)
    return m


def inverse_transform(images):
    return (images + 1.) / 2

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            
    
    
    