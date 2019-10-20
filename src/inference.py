from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf
import config
#from tf_data_handler import inputs

class cnn_model_struct:
    def __init__(self, trainable=False):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self,item)

    # def build(self, patch):
    #
    #     #print ("building the network")
    #     input_patch = tf.identity(patch,name="input_patch")
    #     with tf.name_scope('reshape'):
    #         x_image = tf.reshape(input_patch, [-1, 416, 416, 3])
    #
    #     # conv layer 1
    #     with tf.name_scope('conv1'):
    #         self.W_conv1 = self.weight_variable([3, 3, 3, 16],var_name='wconv1')
    #         self.b_conv1 = self.bias_variable([16],var_name='bconv1')
    #         self.norm1 = tf.layers.batch_normalization(self.conv2d(x_image, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1,scale=True,center=True)
    #         self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
    #
    #     # Pooling layer - downsamples by 2X.
    #     with tf.name_scope('pool1'):
    #         self.h_pool1 = self.max_pool_2x2(self.h_conv1)
    #
    #     # conv layer 2 -- maps 16 feature maps to 32.
    #     with tf.name_scope('conv2'):
    #         self.W_conv2 = self.weight_variable([3, 3, 16, 32],var_name='wconv2')
    #         self.b_conv2 = self.bias_variable([32],var_name='bconv2')
    #         self.norm2 = tf.layers.batch_normalization(self.conv2d(self.h_pool1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2,scale=True,center=True)
    #         self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool2'):
    #         self.h_pool2 = self.max_pool_2x2(self.h_conv2)
    #
    #     # conv layer 3 -- maps 32 feature maps to 64.
    #     with tf.name_scope('conv3'):
    #         self.W_conv3 = self.weight_variable([3, 3, 32, 64],var_name='wconv3')
    #         self.b_conv3 = self.bias_variable([64],var_name='bconv3')
    #         self.norm3 = tf.layers.batch_normalization(self.conv2d(self.h_pool2, self.W_conv3, stride=[1, 1, 1, 1]) + self.b_conv3, scale=True,center=True)
    #         self.h_conv3 = tf.nn.leaky_relu(self.norm3,alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool3'):
    #         self.h_pool3 = self.max_pool_2x2(self.h_conv3)
    #
    #     # conv layer 4 -- maps 64 feature maps to 128.
    #     with tf.name_scope('conv4'):
    #         self.W_conv4 = self.weight_variable([3, 3, 64, 128],var_name='wconv4')
    #         self.b_conv4 = self.bias_variable([128],var_name='bconv4')
    #         self.norm4 = tf.layers.batch_normalization(self.conv2d(self.h_pool3, self.W_conv4, stride=[1, 1, 1, 1]) + self.b_conv4, scale=True,center=True)
    #         self.h_conv4 = tf.nn.leaky_relu(self.norm4, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool4'):
    #         self.h_pool4 = self.max_pool_2x2(self.h_conv4)
    #
    #     # conv layer 5 -- maps 128 feature maps to 256.
    #     with tf.name_scope('conv5'):
    #         self.W_conv5 = self.weight_variable([3, 3, 128, 256],var_name='wconv5')
    #         self.b_conv5 = self.bias_variable([256],var_name='bconv5')
    #         self.norm5 = tf.layers.batch_normalization(self.conv2d(self.h_pool4, self.W_conv5, stride=[1, 1, 1, 1]) + self.b_conv5, scale=True,center=True)
    #         self.h_conv5 = tf.nn.leaky_relu(self.norm5, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool5'):
    #         self.h_pool5 = self.max_pool_2x2(self.h_conv5)
    #
    #     # conv layer 6 -- maps 32 feature maps to 64.
    #     with tf.name_scope('conv6'):
    #         self.W_conv6 = self.weight_variable([3, 3, 256, 512],var_name='wconv6')
    #         self.b_conv6 = self.bias_variable([512],var_name='bconv6')
    #         self.norm6 = tf.layers.batch_normalization(self.conv2d(self.h_pool5, self.W_conv6, stride=[1, 1, 1, 1]) + self.b_conv6, scale=True,center=True)
    #         self.h_conv6 = tf.nn.leaky_relu(self.norm6,alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool6'):
    #         self.h_pool6 = self.max_pool_2x2_1(self.h_conv6)
    #
    #     # conv layer 7 -- 512 to 1024
    #     with tf.name_scope('conv7'):
    #         self.W_conv7 = self.weight_variable([3, 3, 512, 1024],var_name='wconv7')
    #         self.b_conv7 = self.bias_variable([1024],var_name='bconv7')
    #         self.norm7 = tf.layers.batch_normalization(self.conv2d(self.h_pool6, self.W_conv7,stride=[1,1,1,1]) + self.b_conv7, scale=True,center=True)
    #         self.h_conv7 = tf.nn.leaky_relu(self.norm7, alpha=0.1)
    #
    #     # conv layer 8 -- 1024 to 1024
    #     with tf.name_scope('conv8'):
    #         self.W_conv8 = self.weight_variable([3, 3, 1024, 1024],var_name='wconv8')
    #         self.b_conv8 = self.bias_variable([1024],var_name='bconv8')
    #         self.norm8 = tf.layers.batch_normalization(self.conv2d(self.h_conv7, self.W_conv8,stride=[1,1,1,1]) + self.b_conv8, scale=True,center=True)
    #         self.h_conv8 = tf.nn.leaky_relu(self.norm8, alpha=0.1)
    #
    #     # conv layer 9 -- 1x1 conv for the output
    #     with tf.name_scope('conv9'):
    #         self.W_conv9 = self.weight_variable([1, 1, 1024, 5],var_name='wconv9')
    #         self.b_conv9 = self.bias_variable([5],var_name='bconv9')
    #         #self.h_conv9 = tf.nn.relu(self.conv2d(self.h_conv8, self.W_conv9,stride=[1,1,1,1]) + self.b_conv9)
    #         self.h_conv9 = self.conv2d(self.h_conv8, self.W_conv9, stride=[1, 1, 1, 1]) + self.b_conv9
    #
    #     self.output = tf.identity(self.h_conv9,name="output")
    #     return self.output


    # def build(self, patch, train_mode=False):
    #
    #     #print ("building the network")
    #     input_patch = tf.identity(tf.image.resize_images(patch, [416,416], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),name="input_patch")
    #     with tf.name_scope('reshape'):
    #         x_image = tf.reshape(input_patch, [-1, 416, 416, 3])
    #
    #     # conv layer 1
    #     with tf.name_scope('conv1'):
    #         self.W_conv1 = self.weight_variable([3, 3, 3, 16],var_name='wconv1')
    #         self.b_conv1 = self.bias_variable([16],var_name='bconv1')
    #         #if train_mode:
    #         self.norm1 = tf.layers.batch_normalization(self.conv2d(x_image, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1,scale=True,center=True,training=train_mode)
    #         self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
    #         #else:
    #         #    self.h_conv1 = tf.nn.leaky_relu(self.conv2d(x_image, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1, alpha=0.1)
    #
    #     # Pooling layer - downsamples by 2X.
    #     with tf.name_scope('pool1'):
    #         self.h_pool1 = self.max_pool_2x2(self.h_conv1)
    #
    #     # conv layer 2 -- maps 16 feature maps to 32.
    #     with tf.name_scope('conv2'):
    #         self.W_conv2 = self.weight_variable([3, 3, 16, 32],var_name='wconv2')
    #         self.b_conv2 = self.bias_variable([32],var_name='bconv2')
    #         #if train_mode:
    #         self.norm2 = tf.layers.batch_normalization(self.conv2d(self.h_pool1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2,scale=True,center=True,training=train_mode)
    #         self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
    #         #else:
    #         #    self.h_conv2 = tf.nn.leaky_relu(
    #         #        self.conv2d(self.h_pool1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool2'):
    #         self.h_pool2 = self.max_pool_2x2(self.h_conv2)
    #
    #     # conv layer 3 -- maps 32 feature maps to 64.
    #     with tf.name_scope('conv3'):
    #         self.W_conv3 = self.weight_variable([3, 3, 32, 64],var_name='wconv3')
    #         self.b_conv3 = self.bias_variable([64],var_name='bconv3')
    #
    #         #if train_mode:
    #         self.norm3 = tf.layers.batch_normalization(self.conv2d(self.h_pool2, self.W_conv3, stride=[1, 1, 1, 1]) + self.b_conv3, scale=True,center=True,training=train_mode)
    #         self.h_conv3 = tf.nn.leaky_relu(self.norm3,alpha=0.1)
    #         #else:
    #         #    self.h_conv3 = tf.nn.leaky_relu(
    #         #        self.conv2d(self.h_pool2, self.W_conv3, stride=[1, 1, 1, 1]) + self.b_conv3, alpha=0.1)
    #     # Second pooling layer.
    #     with tf.name_scope('pool3'):
    #         self.h_pool3 = self.max_pool_2x2(self.h_conv3)
    #
    #     # conv layer 4 -- maps 64 feature maps to 128.
    #     with tf.name_scope('conv4'):
    #         self.W_conv4 = self.weight_variable([3, 3, 64, 128],var_name='wconv4')
    #         self.b_conv4 = self.bias_variable([128],var_name='bconv4')
    #         #if train_mode:
    #         self.norm4 = tf.layers.batch_normalization(self.conv2d(self.h_pool3, self.W_conv4, stride=[1, 1, 1, 1]) + self.b_conv4, scale=True,center=True,training=train_mode)
    #         self.h_conv4 = tf.nn.leaky_relu(self.norm4, alpha=0.1)
    #         #else:
    #         #    self.h_conv4 = tf.nn.leaky_relu(
    #         #        self.conv2d(self.h_pool3, self.W_conv4, stride=[1, 1, 1, 1]) + self.b_conv4, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool4'):
    #         self.h_pool4 = self.max_pool_2x2(self.h_conv4)
    #
    #     # conv layer 5 -- maps 128 feature maps to 256.
    #     with tf.name_scope('conv5'):
    #         self.W_conv5 = self.weight_variable([3, 3, 128, 256],var_name='wconv5')
    #         self.b_conv5 = self.bias_variable([256],var_name='bconv5')
    #         #if train_mode:
    #         self.norm5 = tf.layers.batch_normalization(self.conv2d(self.h_pool4, self.W_conv5, stride=[1, 1, 1, 1]) + self.b_conv5, scale=True,center=True,training=train_mode)
    #         self.h_conv5 = tf.nn.leaky_relu(self.norm5, alpha=0.1)
    #         #else:
    #         #    self.h_conv5 = tf.nn.leaky_relu(
    #         #        self.conv2d(self.h_pool4, self.W_conv5, stride=[1, 1, 1, 1]) + self.b_conv5, alpha=0.1)
    #     # Second pooling layer.
    #     with tf.name_scope('pool5'):
    #         self.h_pool5 = self.max_pool_2x2(self.h_conv5)
    #
    #     # conv layer 6 -- maps 32 feature maps to 64.
    #     with tf.name_scope('conv6'):
    #         self.W_conv6 = self.weight_variable([3, 3, 256, 512],var_name='wconv6')
    #         self.b_conv6 = self.bias_variable([512],var_name='bconv6')
    #
    #         #if train_mode:
    #         self.norm6 = tf.layers.batch_normalization(self.conv2d(self.h_pool5, self.W_conv6, stride=[1, 1, 1, 1]) + self.b_conv6, scale=True,center=True,training=train_mode)
    #         self.h_conv6 = tf.nn.leaky_relu(self.norm6,alpha=0.1)
    #         #else:
    #         #    self.h_conv6 = tf.nn.leaky_relu(
    #         #        self.conv2d(self.h_pool5, self.W_conv6, stride=[1, 1, 1, 1]) + self.b_conv6, alpha=0.1)
    #
    #     # Second pooling layer.
    #     with tf.name_scope('pool6'):
    #         self.h_pool6 = self.max_pool_2x2_1(self.h_conv6)
    #
    #     # conv layer 9 -- 1x1 conv for the output
    #     with tf.name_scope('conv7'):
    #         self.W_conv7 = self.weight_variable([1, 1, 512, 3],var_name='wconv7')
    #         self.b_conv7 = self.bias_variable([3],var_name='bconv7')
    #         self.h_conv7 = self.conv2d(self.h_pool6, self.W_conv7, stride=[1, 1, 1, 1]) + self.b_conv7
    #
    #     self.output = tf.identity(self.h_conv7,name="output")

    def build(self, patch, train_mode=False):

        print ("building the network")
        input_patch = tf.identity(patch,name="input_patch")
        with tf.name_scope('reshape'):
            x_image = tf.reshape(input_patch, [-1, 416, 416, 1])

        # conv layer 1
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_variable([3, 3, 1, 8],var_name='wconv1')
            self.b_conv1 = self.bias_variable([8],var_name='bconv1')
            #if train_mode:
            self.norm1 = tf.layers.batch_normalization(self.conv2d(x_image, self.W_conv1,stride=[1,2,2,1]) + self.b_conv1,scale=True,center=True,training=train_mode)
            self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
            #else:
            #    self.h_conv1 = tf.nn.leaky_relu(self.conv2d(x_image, self.W_conv1,stride=[1,1,1,1]) + self.b_conv1, alpha=0.1)

        # # Pooling layer - downsamples by 2X.
        # with tf.name_scope('pool1'):
        #     self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # conv layer 2 -- maps 16 feature maps to 32.
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_variable([3, 3, 8, 16],var_name='wconv2')
            self.b_conv2 = self.bias_variable([16],var_name='bconv2')
            #if train_mode:
            self.norm2 = tf.layers.batch_normalization(self.conv2d(self.h_conv1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2,scale=True,center=True,training=train_mode)
            self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
            #else:
            #    self.h_conv2 = tf.nn.leaky_relu(
            #        self.conv2d(self.h_pool1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2, alpha=0.1)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # conv layer 3 -- maps 32 feature maps to 64.
        with tf.name_scope('conv3'):
            self.W_conv3 = self.weight_variable([3, 3, 16, 32],var_name='wconv3')
            self.b_conv3 = self.bias_variable([32],var_name='bconv3')

            #if train_mode:
            self.norm3 = tf.layers.batch_normalization(self.conv2d(self.h_pool2, self.W_conv3, stride=[1, 2, 2, 1]) + self.b_conv3, scale=True,center=True,training=train_mode)
            self.h_conv3 = tf.nn.leaky_relu(self.norm3,alpha=0.1)
            #else:
            #    self.h_conv3 = tf.nn.leaky_relu(
            #        self.conv2d(self.h_pool2, self.W_conv3, stride=[1, 1, 1, 1]) + self.b_conv3, alpha=0.1)
        # Second pooling layer.
        # with tf.name_scope('pool3'):
        #     self.h_pool3 = self.max_pool_2x2(self.h_conv3)

        # conv layer 4 -- maps 64 feature maps to 128.
        with tf.name_scope('conv4'):
            self.W_conv4 = self.weight_variable([3, 3, 32, 64],var_name='wconv4')
            self.b_conv4 = self.bias_variable([64],var_name='bconv4')
            #if train_mode:
            self.norm4 = tf.layers.batch_normalization(self.conv2d(self.h_conv3, self.W_conv4, stride=[1, 1, 1, 1]) + self.b_conv4, scale=True,center=True,training=train_mode)
            self.h_conv4 = tf.nn.leaky_relu(self.norm4, alpha=0.1)
            #else:
            #    self.h_conv4 = tf.nn.leaky_relu(
            #        self.conv2d(self.h_pool3, self.W_conv4, stride=[1, 1, 1, 1]) + self.b_conv4, alpha=0.1)

        # Second pooling layer.
        with tf.name_scope('pool4'):
            self.h_pool4 = self.max_pool_2x2(self.h_conv4)

        # conv layer 5 -- maps 128 feature maps to 256.
        with tf.name_scope('conv5'):
            self.W_conv5 = self.weight_variable([3, 3, 64, 128],var_name='wconv5')
            self.b_conv5 = self.bias_variable([128],var_name='bconv5')
            #if train_mode:
            self.norm5 = tf.layers.batch_normalization(self.conv2d(self.h_pool4, self.W_conv5, stride=[1, 2, 2, 1]) + self.b_conv5, scale=True,center=True,training=train_mode)
            self.h_conv5 = tf.nn.leaky_relu(self.norm5, alpha=0.1)
            #else:
            #    self.h_conv5 = tf.nn.leaky_relu(
            #        self.conv2d(self.h_pool4, self.W_conv5, stride=[1, 1, 1, 1]) + self.b_conv5, alpha=0.1)
        # Second pooling layer.
        # with tf.name_scope('pool5'):
        #     self.h_pool5 = self.max_pool_2x2(self.h_conv5)

        # conv layer 6 -- maps 32 feature maps to 64.
        with tf.name_scope('conv6'):
            self.W_conv6 = self.weight_variable([3, 3, 128, 256],var_name='wconv6')
            self.b_conv6 = self.bias_variable([256],var_name='bconv6')

            #if train_mode:
            self.norm6 = tf.layers.batch_normalization(self.conv2d(self.h_conv5, self.W_conv6, stride=[1, 1, 1, 1]) + self.b_conv6, scale=True,center=True,training=train_mode)
            self.h_conv6 = tf.nn.leaky_relu(self.norm6,alpha=0.1)
            #else:
            #    self.h_conv6 = tf.nn.leaky_relu(
            #        self.conv2d(self.h_pool5, self.W_conv6, stride=[1, 1, 1, 1]) + self.b_conv6, alpha=0.1)

        # Second pooling layer.
        with tf.name_scope('pool6'):
            self.h_pool6 = self.max_pool_2x2_1(self.h_conv6)

        # conv layer 9 -- 1x1 conv for the output
        with tf.name_scope('conv7'):
            self.W_conv7 = self.weight_variable([1, 1, 256, 3],var_name='wconv7')
            self.b_conv7 = self.bias_variable([3],var_name='bconv7')
            self.h_conv7 = self.conv2d(self.h_pool6, self.W_conv7, stride=[1, 1, 1, 1]) + self.b_conv7

        self.output = tf.identity(self.h_conv7,name="output")

    def conv2d(self, x, W, stride=[1,1,1,1]):
        """conv2d returns a 2d convolution layer with full stride."""
        #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2_1(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')

    def weight_variable(self, shape, var_name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name=var_name,initializer=initial)

    def bias_variable(self, shape, var_name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.001, shape=shape)
        return tf.get_variable(name=var_name,initializer=initial)

def model_inference():
    x = tf.placeholder(tf.float32,[None, 416, 416, 1],name='patch')
    #x = tf.placeholder(tf.float32, [None, 1920, 1080, 3], name='patch')
    with tf.variable_scope("model") as scope:
        model = cnn_model_struct()
        yconv = model.build(x)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #saver.restore(sess,'/media/data_cifs/lakshmi/zebrafish/summaries/cnn_box_30750.ckpt-30750')
        #saver.save(sess,'/media/data_cifs/lakshmi/zebrafish/bounding_box_inference_v2')
        saver.restore(sess,'/media/data_cifs/lakshmi/zebrafish/darkAndLight_Bootstrapped/cnn_box_60000.ckpt-60000')
        saver.save(sess,'/media/data_cifs/lakshmi/zebrafish/inference_640x480_darkAndLightBootstrapped')

if __name__ == '__main__':
    #import ipdb; ipdb.set_trace();
    model_inference()