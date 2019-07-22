# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from glob import glob
import cv2
import csv
import sys

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *


parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    kwargs = {'newline': ''}
    mode = 'w'
    if sys.version_info < (3, 0):
        kwargs.pop('newline', None)
        mode = 'wb'

    with open(args.image_path+'predictions.csv', mode, **kwargs) as fp:
        writer = csv.writer(fp, delimiter=',')
        # writer.writerow(["your", "header", "foo"])  # write header

        for path_ in glob(args.image_path+"frame*"):
        
            input_image = scipy.misc.imread(path_, mode="RGB")
            original_height, original_width, num_channels = input_image.shape
            input_image_ = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
            input_image_ = input_image_.astype(np.float32) / 255
            input_images = np.stack((input_image_, np.fliplr(input_image_)), 0)

            # Predict
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

            # output_directory = os.path.dirname("./output/samples"+"_"+restore_path.split('/')[-1]+"/")
            ## output_name = os.path.splitext(os.path.basename(path_.split('/')[0]))[0]
            #output_name = path_.split('/')[-1]

            # np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
            disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
            #plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

            # img = disp_to_img[:,:,2].flatten()
            blur = cv2.blur(disp_to_img,(10,10))
            img = blur.flatten()
            
            mask = (img<np.percentile(img, 20)).astype(int)
            mask = mask>0
            mask = mask.reshape((disp_to_img.shape))
            img_ = np.zeros((720, 1280, 3))
            img_[:,:,2] = mask
            plt.imsave("mask.png", img_, cmap='plasma')
            ret, thresh = cv2.threshold( (mask).astype('uint8'), 0, 1, 0)
            contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # print (contours)
            # print (len(contours))

            max_area = 0
            max_cnt = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # print (area)
                if area>max_area:
                    max_area = area
                    max_cnt = cnt.copy()

            # if max_cnt!=0:
            mask = np.zeros((disp_to_img.shape))
            cv2.drawContours(mask,[max_cnt],0,1,-1)

            # img_ = np.zeros((720, 1280, 3))
            # img_[:,:,2] = mask
            # plt.imsave("mask2.png", img_, cmap='plasma')

            # print (np.nonzero(mask))
            
            points = np.nonzero(mask)[1]
            pathMid = int(np.ceil(np.mean(points)))

            filename_ = path_.split('frame')
            # depth_img_path = filename_[0]+'depth'+filename_[1]

            writer.writerow([ filename_[1].split('.')[0], pathMid])

    print('done!')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
