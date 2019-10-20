import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import progressbar
import tensorflow as tf
from train_detector import cnn_model_struct
import skvideo.io

class Tester:
    def __init__(self,config):
        self.config = config
        self.input = tf.placeholder(tf.float32,
                            [None,config.image_target_size[0],config.image_target_size[1],config.image_target_size[2]], name='ip_placeholder')
        self.initialized = False

        with tf.device('/gpu:0'):
            with tf.variable_scope("model") as scope:
                self.model = cnn_model_struct()
                self.model.build(self.input, config.num_classes, train_mode=False)

            self.gpuconfig = tf.ConfigProto()
            self.gpuconfig.gpu_options.allow_growth = True
            self.gpuconfig.allow_soft_placement = True
            self.saver = tf.train.Saver()

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self,item)

    def make_predictions(self, patches):
        try:
            probs = []
            if self.initialized == False:
                self.sess = tf.Session(config=self.gpuconfig)
                ckpts = tf.train.latest_checkpoint(self.config.model_output)
                print ckpts
                self.saver.restore(self.sess, ckpts)
                self.initialized = True
            probs = self.sess.run(self.model.output,feed_dict={self.input:patches})

            # with tf.Session(config=self.gpuconfig) as sess:
            #     if self.initialized == False:
            #         ckpts = tf.train.latest_checkpoint(self.config.model_output)
            #         self.saver.restore(sess, ckpts)
            #         self.initialized = True
            #     probs = sess.run(self.model.output,feed_dict={self.input:patches})
        except tf.errors.NotFoundError:
            print ('checkpoint could not be restored')
        finally:
            return probs

def get_coordinates(output_maps):
    #import ipdb; ipdb.set_trace()
    X = output_maps.squeeze()[:,:,-1]
    idx = np.unravel_index(np.argmax(X),X.shape)
    #idx = [9,8]
    vals = output_maps[0,idx[0],idx[1],:]
    v1 = idx[0] * 32 + vals[0] * 32
    v2 = idx[1] * 32 + vals[1] * 32
    #return np.array([idx[1]*32,idx[0]*32])
    return np.array([v1, v2])


def process(video_name,tester,config):

    # Load in the video to read
    #video_stream = cv2.VideoCapture(video_name)
    video_stream = skvideo.io.vreader(video_name)

    frameid = 0
    max_lim = 8995
    bar = progressbar.ProgressBar(max_value=max_lim)
    #bar = progressbar.ProgressBar(max_lim)
    bar.start()

    print 'Reading from video...'

    fig, ax = plt.subplots(1)
    Xloc, Yloc = [], []

    #while (video_stream.isOpened()):
    #    ret, frame = video_stream.read()
    for frame in video_stream:
        if frameid < 10: #100
            frameid = frameid + 1
            continue

        if frameid > max_lim:
            break

        # get the current frame
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        #Y, U, V = cv2.split(rgb_frame)
        Y, U, V = cv2.split(frame)
        rgb_frame = Y

        if config.resize_ims:
            rgb_frame = cv2.resize(rgb_frame, (config.image_target_size[0], config.image_target_size[1]))

        frame = rgb_frame / 255. - 0.5
        frame = np.expand_dims(frame, 2)

        #import ipdb; ipdb.set_trace();
        output_values = tester.make_predictions(patches=[frame])
        coords = get_coordinates(output_values)
        #plt.clf(); plt.imshow(rgb_frame); plt.scatter(coords[1],coords[0]); plt.pause(0.001);

        # ax.clear()
        # ax.imshow(rgb_frame)
        # rect = mpl.patches.Rectangle((coords[1]-16,coords[0]-16),32,32, linewidth=1,edgecolor='r',facecolor='none')
        # ax.add_patch(rect)
        # plt.pause(0.1)
        # #plt.show()

        v1 = coords[0] * 480. / 416.
        v2 = coords[1] * 640. / 416.
        #Xloc.append( v2 * (-0.5779) + 192.3)
        #Yloc.append( v1 * 0.5870 - 120.1)

        Xloc.append( v2 * (-0.5673) + 186.5)
        Yloc.append( v1 * 0.5722 - 107.1)

        frameid = frameid + 1
        bar.update(frameid)
        if frameid == max_lim:
            break

    #np.save('jnt_errors',err)
    bar.finish()

    plt.scatter(Xloc,Yloc,2,c=np.arange(len(Xloc)),cmap='jet',linewidths=0)
    plt.xlim([-150, 150])
    plt.ylim([-150, 150])
    plt.savefig('/media/data_cifs/lakshmi/zebrafish/analysis/'+video_name.split('/')[-1][:-3]+'png')
    np.savetxt('/media/data_cifs/lakshmi/zebrafish/analysis/'+video_name.split('/')[-1][:-3]+'txt',np.array([Xloc,Yloc]))

    #print 'Finished...'

def eval_video_with_model(config=None):

    tester = Tester(config=config)
    for video in config.test_video_name:
        process(video,tester,config)
    os._exit(0)