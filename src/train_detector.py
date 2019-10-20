from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf
import config
from tf_data_handler import inputs
import os

class cnn_model_struct:
    def __init__(self, trainable=False):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self,item)

    def build(self, patch, output_shape,train_mode=None):

        print ("building the network")
        input_patch = tf.identity(patch,name="input_patch")
        with tf.name_scope('reshape'):
            x_image = tf.reshape(input_patch, [-1, 416, 416, 1]) # 3 for rgb, 1 for Y

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

def calc_error(labels,predictions):
    # note that here there are only 2 dimensions -- batch index and (u,v,d)
    assert (labels.shape == predictions.shape)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-predictions), 1)),0)

def train_model(config):

    with tf.device('/cpu:0'):
        train_data = tf.placeholder(tf.float32, config.param_dims)
        train_labels = tf.placeholder(tf.float32, config.output_hist_dims) 

    import ipdb; ipdb.set_trace()
    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            print ("creating the model")
            model = cnn_model_struct()
            model.build(train_images,config.num_classes,train_mode=True)
            y_conv = model.output

            # Define loss and optimizer
            with tf.name_scope('loss'):
                reg_loss = tf.nn.l2_loss(y_conv - train_labels)

            with tf.name_scope('adam_optimizer'):
                # wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
                # loss_wd = reg_loss+(0.0005 * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
                # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_wd)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(reg_loss)

            # with tf.name_scope('accuracy'):
            #     res_shaped = tf.reshape(y_conv, [config.train_batch, config.num_classes])
            #     lab_shaped = tf.reshape(train_labels, [config.train_batch, config.num_classes])
            # accuracy = calc_error(lab_shaped, res_shaped)

            print("using validation")
            # scope.reuse_variables()
            with tf.variable_scope('val_model', reuse=tf.AUTO_REUSE):
                val_model = cnn_model_struct()
                val_model.build(val_images, config.num_classes, train_mode=False)
                val_res = val_model.output
                # val_res_shaped = tf.reshape(val_model.output, [config.val_batch, config.num_classes])
                # val_lab_shaped = tf.reshape(val_labels, [config.val_batch, config.num_classes])
                val_error =  tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(val_labels-val_res))))

            tf.summary.scalar("loss", reg_loss)
            #tf.summary.scalar("train error", accuracy)
            #tf.summary.scalar("validation error", val_error)
            summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement = True

    with tf.Session(config=gpuconfig) as sess:
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        try:
            while not coord.should_stop():
                # train for a step
                _, tr_images, tr_labels, loss, softmax_outputs = sess.run([train_step,train_images,train_labels, reg_loss, y_conv])
                print("step={}, loss={}".format(step,loss))
                step+=1
                #import ipdb; ipdb.set_trace()

                # validating the model. main concern is if the weights are shared between
                # the train and validation model
                if step % 200 == 0:
                    vl_img, vl_lab, vl_res, vl_err = sess.run([val_images,val_labels,val_res,val_error])
                    print("\t validating")
                    print("\t val error = {}".format(vl_err))

                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str,step)
                # save the model check point
                if step % 250 == 0:
                    saver.save(sess,os.path.join(
                        config.model_output,
                        config.model_name+'_'+str(step)+'.ckpt'
                    ),global_step=step)

        except tf.errors.OutOfRangeError:
            print("Finished training for %d epochs" % config.epochs)
        finally:
            coord.request_stop()
            coord.join(threads)

def test_model_eval(config):
    test_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    with tf.device('/cpu:0'):
        test_images, test_labels = inputs(tfrecord_file=test_data,
                                            num_epochs=None,
                                            image_target_size=config.image_target_size,
                                            label_shape=config.num_classes,
                                            batch_size=config.test_batch,
                                            augmentation=False)

    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            model = cnn_model_struct()
            model.build(test_images,config.num_classes,train_mode=False)
            results = tf.argmax(model.output, 1)
            error = tf.reduce_mean(tf.cast(tf.equal(results, tf.cast(test_labels, tf.int64)), tf.float32))

        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig) as sess:
            #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            #sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            step=0
            try:
                while not coord.should_stop():
                    # load the model here
                    ckpts=tf.train.latest_checkpoint(config.model_output)
                    saver.restore(sess,ckpts)
                    ims, labs, probs, err, res = sess.run([test_images,test_labels,model.output,error,results])
                    import ipdb; ipdb.set_trace();
            except tf.errors.OutOfRangeError:
                print('Epoch limit reached!')
            finally:
                coord.request_stop()
            coord.join(threads)

# def get_model_predictions(config,patches):
#     input = tf.placeholder(tf.float32, [None,config.image_target_size[0],config.image_target_size[1],config.image_target_size[2]], name='ip_placeholder')
#     with tf.device('/gpu:0'):
#         with tf.variable_scope("model") as scope:
#             model = cnn_model_struct()
#             model.build(input,config.num_classes,train_mode=False)
#
#         gpuconfig = tf.ConfigProto()
#         gpuconfig.gpu_options.allow_growth = True
#         gpuconfig.allow_soft_placement = True
#         saver = tf.train.Saver()
#
#         with tf.Session(config=gpuconfig) as sess:
#             #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#             #sess.run(init_op)
#             #coord = tf.train.Coordinator()
#             #threads = tf.train.start_queue_runners(coord=coord)
#             step=0
#             try:
#                 #while not coord.should_stop():
#                     # load the model here
#                     ckpts=tf.train.latest_checkpoint(config.model_output)
#                     saver.restore(sess,ckpts)
#                     probs = sess.run(model.output,feed_dict={input:patches})
#             except tf.errors.OutOfRangeError:
#                 print('Epoch limit reached!')
#             finally:
#                 #coord.request_stop()
#                 print ('done')
#             #coord.join(threads)
#     return probs
