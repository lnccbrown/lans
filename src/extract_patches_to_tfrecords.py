import cv2
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from tf_data_handler import write_tfrecord
import config
import progressbar


'''
just get the center of the animal body from the annotations
'''
def get_fish_location(
        frameid,
        locations
):
    # for now, consider oly the first object
    obj = 0
    # just focus on the head
    jnt = 0
    y, x = int(locations[obj][jnt][frameid, 0]), int(locations[obj][jnt][frameid, 1])
    return y, x
"""
Extract local regions around a landmark and give the appropriate label
Sample some from the background as well in the process
"""


def make_image_patches(
        rgb_frame,
        frameid,
        locations,
        objects_to_include=[],
        joints=[0],
        patch_size=[28, 28]
):
    patches = []
    labels = []

    for o in range(len(objects_to_include)):
        obj = objects_to_include[o]
        for j in range(len(joints)):
            #jnt = joints[j]
            jnt = j
            y, x = int(locations[obj][jnt][frameid, 0]), int(locations[obj][jnt][frameid, 1])

            # get the positive example
            patch = rgb_frame[(x - patch_size[0] / 2 - 1):(x + patch_size[0] / 2 - 1),
                    (y - patch_size[1] / 2 - 1):(y + patch_size[1] / 2 - 1), :]
            patches.append(patch.astype(np.float32))
            labels.append(np.float32(jnt + 1))

            #import ipdb; ipdb.set_trace()
            #if jnt == 0:
            #    plt.imshow(patch)
            #    plt.show()
            # also get a corrsponding negative example

            if jnt in [1,2,3]:
                random_noise = np.random.randint(30, size=3)
                bg_patch = rgb_frame[
                       (x + random_noise[0] - patch_size[0] / 2 - 1):(x + random_noise[0] + patch_size[0] / 2 - 1),
                       (y + random_noise[1] - patch_size[1] / 2 - 1):(y + random_noise[1] + patch_size[1] / 2 - 1), :]
                #plt.imshow(bg_patch); plt.show();
                patches.append(bg_patch.astype(np.float32))
                labels.append(np.float32(0))
    return patches, labels


"""
load the labels obtained from running Yuliang's tacking system
return parameter: 
   [object][joint][frame,x/y]
"""


def load_joints(video_name,
                label_folder,
                objects_to_include=[],
                joints=[0, 1]):
    assert len(objects_to_include) > 0, 'No objects specified'
    jnts = []

    for object in list(objects_to_include):
        lab = scio.loadmat(os.path.join(label_folder, '%s_obj_%d_seq.mat' % (video_name.split('.')[0], object + 1)))

        jnt_list = []
        for joint in joints:
            xy_locs = np.transpose(np.asarray([lab['x_seq'][joint, :], lab['y_seq'][joint, :]]))
            jnt_list.append(xy_locs)
        jnts.append(jnt_list)
    return jnts

def load_joints_annotations(video_name,
                label_folder,
                objects_to_include=[],
                joints=[0, 1]):
    assert len(objects_to_include) > 0, 'No objects specified'
    jnts = []

    for object in list(objects_to_include):
        lab = scio.loadmat(os.path.join(label_folder, '%s_obj_%d_seq.mat' % (video_name.split('.')[0], object)))

        jnt_list = []
        for joint in joints:
            #import ipdb; ipdb.set_trace()
            if 'xseq' in lab.keys():
                xy_locs = np.transpose(np.asarray([lab['xseq'][joint, :], lab['yseq'][joint, :]]))
            else:
                xy_locs = np.transpose(np.asarray([lab['x_seq'][joint, :], lab['y_seq'][joint, :]]))

            jnt_list.append(xy_locs)
        jnts.append(jnt_list)
    return jnts

def make_labels(
        frame,
        frameid,
        locations,
        config
):
    #import ipdb; ipdb.set_trace();
    labels = np.zeros((config.label_shape[0],config.label_shape[1],config.label_shape[2]))
    y, x = get_fish_location(frameid=frameid,
                             locations=locations)
    # rescale this coordinates to the rescaled image
    y = y * (config.image_target_size[1] / float(config.image_orig_size[1]))
    x = x * (config.image_target_size[0] / float(config.image_orig_size[0]))

    #plt.clf(); plt.imshow(frame); plt.scatter(y,x); plt.pause(0.001);

    bin_size_x = int(config.image_target_size[0] / config.label_shape[0])
    bin_size_y = int(config.image_target_size[1] / config.label_shape[1])

    bin_x = int(x / bin_size_x)
    bin_y = int(y / bin_size_y)

    off_x = (x - (bin_x * bin_size_x)) / float(bin_size_x)
    off_y = (y - (bin_y * bin_size_y)) / float(bin_size_y)

    # here using a bounding box of size 64x64
    #labels[bin_x,bin_y,:] = np.array([off_x,off_y,64/float(config.image_target_size[0]),64/float(config.image_target_size[0]),1])
    labels[bin_x, bin_y, :] = np.array(
        [off_x, off_y, 1.])
    return labels

def main(
        video_folder='',
        videos='',
        label_folder='',
        objects_to_include=None,
        tfrecord_dir='',
        config=None
):
    data_prop = config.data_prop
    all_patches, all_labels = [], []

    frame_lower_limit = 10

    for video_name in videos:
        jnts = load_joints_annotations(video_name=video_name,
                           label_folder=label_folder,
                           objects_to_include=objects_to_include,
                           joints=config.joints_to_extract)

        frame_upper_limit = len(jnts[0][0][:,0])-1
        # Load in the video to read
        video_stream = cv2.VideoCapture(os.path.join(video_folder, video_name))
        frameid = 0
        bar = progressbar.ProgressBar(frame_upper_limit)
        bar.start()

        print 'Reading from video...'
        while (video_stream.isOpened()):
            flag, frame = video_stream.read()
            assert flag, 'Reading from video has failed!'

            if (frameid < frame_lower_limit):
                frameid += 1
                continue

            if (frameid >= frame_upper_limit):
                break

            Y, U, V = cv2.split(frame)

            if config.resize_ims:
                rgb_frame = cv2.resize(Y,(config.image_target_size[0],config.image_target_size[1]))

            # if video_name == 'sub05_video01.mp4':
            labels = make_labels(frame=rgb_frame,
                                     frameid=frameid,
                                     locations=jnts,
                                     config=config)
            # else:
            #     import ipdb; ipdb.set_trace()
            #     labels = make_labels(frame=rgb_frame,
            #             frameid=(frameid-frame_lower_limit),
            #             locations=jnts,
            #             config=config)


            all_patches.extend([rgb_frame.astype(np.float32)])
            all_labels.extend(np.array([labels],dtype=np.float32))

            frameid = frameid + 1
            bar.update(frameid)

        bar.finish()
    print 'Finished reading from video. Making train vs test split'

    #import ipdb; ipdb.set_trace();
    total_items = len(all_patches)
    arr = np.arange(total_items)
    np.random.shuffle(arr)

    all_patches = np.asarray(all_patches)[arr]
    all_labels = np.asarray(all_labels)[arr]
    #assert (np.sum(data_prop.values()) != 1.), 'Train vs Test split specified incorrectly'

    train_idx_lim = int(data_prop['train'] * total_items)
    val_idx_lim = int((data_prop['train'] + data_prop['val']) * total_items)

    # write the tf records as specified
    if config.train_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.train_tfrecords,
            all_patches[:train_idx_lim],
            all_labels[:train_idx_lim])

    if config.val_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.val_tfrecords,
            all_patches[train_idx_lim:val_idx_lim],
            all_labels[train_idx_lim:val_idx_lim])

    if config.test_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.test_tfrecords,
            all_patches[val_idx_lim:],
            all_labels[val_idx_lim:])


if __name__ == '__main__':
    config = config.Config()
    main(
        video_folder=config.base_dir,
        videos=config.video_name,
        label_folder=config.label_dir,
        objects_to_include=np.asarray(config.objects_to_include),
        tfrecord_dir=config.tfrecord_dir,
        config=config
    )
