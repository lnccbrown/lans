from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os, glob, pickle
import tensorflow as tf
import config
from tf_data_handler import inputs
import numpy as np
import tqdm, time
import matplotlib.pyplot as plt
import math

class cnn_model_struct:
    def __init__(self, trainable=True):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self, item)

    def get_size(self, input_data):
        return np.prod([int(x) for x in input_data.get_shape()[1:]])

    def build(self, input_data, input_shape, output_shape, train_mode=None, verbose=True):
	if verbose:
            print ("Building the network...")
        network_input = tf.identity(input_data, name='input')
        with tf.name_scope('reshape'):
            x_data = tf.reshape(network_input, [-1, input_shape[0], input_shape[1], input_shape[2]])
        self.upsample1 = self.fc_layer(x_data, self.get_size(x_data), 16, 'upsample1')
	if verbose:
            print(self.upsample1.get_shape())
        self.upsample2 = self.fc_layer(self.upsample1, self.get_size(self.upsample1), 64, 'upsample2')
	if verbose:
            print(self.upsample2.get_shape())
        self.upsample3 = self.fc_layer(self.upsample2, self.get_size(self.upsample2), 256, 'upsample3')
	if verbose:
            print(self.upsample3.get_shape())
        self.upsample4 = self.fc_layer(self.upsample3, self.get_size(self.upsample3), 1024, 'upsample4')
	if verbose:
            print(self.upsample4.get_shape())

        self.upsample4 = tf.expand_dims(tf.expand_dims(self.upsample4,1),-1)

        # conv layer 1
        with tf.variable_scope('conv1'):
            self.W_conv1 = self.weight_variable([1, 5, 1, 8],var_name='wconv1')
            self.b_conv1 = self.bias_variable([8],var_name='bconv1')
            self.norm1 = tf.layers.batch_normalization(self.conv2d(self.upsample4, self.W_conv1,stride=[1,2,2,1]) + self.b_conv1,scale=True,center=True,training=train_mode)
            self.h_conv1 = tf.nn.leaky_relu(self.norm1, alpha=0.1)
	if verbose:
            print(self.h_conv1.get_shape())

        # conv layer 2
        with tf.variable_scope('conv2'):
            self.W_conv2 = self.weight_variable([1, 5, 8, 4],var_name='wconv2')
            self.b_conv2 = self.bias_variable([4],var_name='bconv2')
            self.norm2 = tf.layers.batch_normalization(self.conv2d(self.h_conv1, self.W_conv2, stride=[1, 1, 1, 1]) + self.b_conv2,scale=True,center=True,training=train_mode)
            self.h_conv2 = tf.nn.leaky_relu(self.norm2, alpha=0.1)
	if verbose:
            print(self.h_conv2.get_shape())

        # conv layer 3
        with tf.variable_scope('conv3'):
            self.W_conv3 = self.weight_variable([1, 5, 4, 2],var_name='wconv3')
            self.b_conv3 = self.bias_variable([2],var_name='bconv3')
            self.norm3 = tf.layers.batch_normalization(self.conv2d(self.h_conv2, self.W_conv3, stride=[1, 1, 1, 1]) + self.b_conv3, scale=True,center=True,training=train_mode)
            self.h_conv3 = tf.nn.leaky_relu(self.norm3,alpha=0.1)
	if verbose:
            print(self.h_conv3.get_shape())

        self.final_layer = self.fc_layer(self.h_conv3, self.get_size(self.h_conv3), np.prod(output_shape), 'final_layer')
        self.final_layer = tf.nn.softmax(self.final_layer)
        self.output = tf.identity(self.final_layer,name='output')
	if verbose:
            print(self.output.get_shape())

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

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolean to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
	    #var = tf.get_variable(name=var_name, initializer=value)

        self.var_dict[(name, idx)] = var

        return var

def calc_error(labels,predictions):
    # note that here there are only 2 dimensions -- batch index and (u,v,d)
    assert (labels.shape == predictions.shape)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels-predictions), 1)),0)

def kl_divergence(p, q): 
    return tf.reduce_sum(p * tf.log(1e-30 + p/q))

def train_model(config):

    train_files = os.path.join(
			config.base_dir,
			config.tfrecord_dir,
			config.train_tfrecords)
    val_files = os.path.join(
			config.base_dir,
			config.tfrecord_dir,
			config.val_tfrecords)

    with tf.device('/cpu:0'): 
	train_data, train_labels = inputs(
					tfrecord_file=train_files,
					num_epochs=config.epochs,
					batch_size=config.train_batch,
					target_data_dims=config.param_dims,
					target_label_dims=config.output_hist_dims)
	val_data, val_labels = inputs(
					tfrecord_file=val_files,
					num_epochs=config.epochs,
					batch_size=config.val_batch,
					target_data_dims=config.param_dims,
					target_label_dims=config.output_hist_dims)

    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            print ("creating the model")
            model = cnn_model_struct()
            model.build(train_data, config.param_dims[1:], config.output_hist_dims[1:],train_mode=True)
            y_conv = model.output

            # Define loss and optimizer
            with tf.name_scope('loss'):
                kl_divergence_loss = kl_divergence(y_conv, tf.reshape(train_labels,[-1,np.prod(config.output_hist_dims[1:])]))

            with tf.name_scope('adam_optimizer'):
                # wd_l = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'biases' not in v.name]
                # loss_wd = reg_loss+(0.0005 * tf.add_n([tf.nn.l2_loss(x) for x in wd_l]))
                # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_wd)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(kl_divergence_loss)

            # with tf.name_scope('accuracy'):
            #     res_shaped = tf.reshape(y_conv, [config.train_batch, config.num_classes])
            #     lab_shaped = tf.reshape(train_labels, [config.train_batch, config.num_classes])
            # accuracy = calc_error(lab_shaped, res_shaped)

	    #####
	    ## VALIDATION
	    #####
            print("building a validation model")
            #with tf.variable_scope('val_model', reuse=tf.AUTO_REUSE):
	    scope.reuse_variables()
            val_model = cnn_model_struct()
            val_model.build(val_data, config.param_dims[1:], config.output_hist_dims[1:],train_mode=False)
            val_res = val_model.output
            val_loss =  kl_divergence(val_res, tf.reshape(val_labels, [-1,np.prod(config.output_hist_dims[1:])]))

	    #img = tf.expand_dims(tf.reshape(train_labels,[-1,32,16]),axis=-1)
            tf.summary.scalar("loss", kl_divergence_loss)
	    #tf.summary.image("groundtruth", img)
	    #tf.summary.histogram("predictions",y_conv)
            #tf.summary.scalar("train error", accuracy)
            #tf.summary.scalar("validation error", val_error)
            summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement = True

    with tf.Session(config=gpuconfig) as sess:
        #graph_location = tempfile.mkdtemp()
        #print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(os.path.join(config.base_dir,config.summary_dir))
        train_writer.add_graph(tf.get_default_graph())

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
	start = time.time()
        try:
            while not coord.should_stop():
                # train for a step
                _, loss, softmax_outputs, tr_data, tr_labels = sess.run([train_step, kl_divergence_loss, y_conv, train_data, train_labels])
                step+=1
		if math.isnan(loss):
			import ipdb; ipdb.set_trace()
                #import ipdb; ipdb.set_trace()
                '''
                # validating the model. main concern is if the weights are shared between
                # the train and validation model
                if step % 200 == 0:
                    vl_img, vl_lab, vl_res, vl_err = sess.run([val_images,val_labels,val_res,val_error])
                    print("\t validating")
                    print("\t val error = {}".format(vl_err))

                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str,step)
                # save the model check point
                '''
                if step % config.print_iters == 0:
		    finish = time.time()
                    print("step={}, loss={}, time_elapsed={} s/step".format(step,loss,(finish-start)/float(config.print_iters)))
		    start = finish
                    saver.save(sess,os.path.join(
                        config.model_output,
                        config.model_name+'_'+str(step)+'.ckpt'
                    ),global_step=step)

		if step % config.val_iters == 0:
		    val_forward_pass_time = time.time()
		    v_data, v_labels, v_res, v_loss = sess.run([val_data, val_labels, val_res, val_loss])

		    summary_str = sess.run(summary_op)
		    train_writer.add_summary(summary_str, step)
		    print("\t val loss = {}, time_elapsed = {}s".format(v_loss, time.time() - val_forward_pass_time))
		    '''
		    for kk in range(1):
			X = v_res[kk].reshape(-1,2); 
			plt.plot(X[:,0],color='r',alpha=0.5); 
			plt.plot(v_labels[kk][:,0],'-.r',alpha=0.5);
			plt.legend(['Model','Groundtruth']) 
			plt.plot(X[:,1],color='b',alpha=0.5); 
			plt.plot(v_labels[kk][:,1],'-.b',alpha=0.5); 
			plt.pause(1);
			plt.clf()
		    '''
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
