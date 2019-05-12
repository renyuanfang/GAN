#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:40:45 2019

@author: fiona06
"""

import tensorflow as tf
import argparse
from utils import check_folder, show_all_variables
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from LSGAN import LSGAN
from LS_GAN import LS_GAN
from WGAN_DIV import WGAN_DIV
from DCGAN import DCGAN
from DRAGAN import DRAGAN

def parse_args():
    desc = 'Tensorflow implementation of GAN collections'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--gan_type', type=str, default='LSGAN',
                        choices=['DCGAN', 'WGAN', 'WGAN_GP', 'WGAN_DIV', 'LSGAN', 'LS_GAN', 'DRAGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='anime', 
                        choices=['mnist', 'fashion-mnist', 'celebA', 'anime'],
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=60000, help='The number pf epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='./training',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory name to save the training logs')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or infer')
    return check_args(parser.parse_args())

def check_args(args):
    #--checkpoint_dir
    check_folder(args.checkpoint_dir)
    #--sample_dir
    check_folder(args.sample_dir)
    #--log_dir
    check_folder(args.log_dir)
    #--epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'
    #--batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'
    #--z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'
    
    return args

def main():    
    #parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    models = [DCGAN, WGAN, WGAN_GP, LSGAN, LS_GAN, WGAN_DIV, DRAGAN]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            sample_dir=args.sample_dir,
                            log_dir=args.log_dir,
                            mode=args.mode)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)
        
        gan.build_model()
        show_all_variables()
        if args.mode == 'train':
            gan.train()
            print(" [*] Training finished!")
        elif args.mode == 'infer':
            gan.infer()
            print(" [*] Infer finished!")

if __name__ == '__main__':
    main()
