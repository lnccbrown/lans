import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import cv2

plt.rcParams["font.family"] = "Times New Roman"
path = '/media/data_cifs/lakshmi/zebrafish/behavior_recordings/paths/'
vid_path = '/media/data_cifs/lakshmi/zebrafish/behavior_recordings/'
sub_ids = ['sub04_video02.mp4.txt']
dist_ids = ['sub04_v2.txt']
vid_ids = ['sub04_video02.mp4']

def loadfile(path_to_file):
    print path_to_file
    X = np.asarray(np.loadtxt(path_to_file, delimiter=','))
    xs = []
    ys = []
    for i in range(X.shape[0]):
        x, y = X[i, 1], X[i, 2]
        if (x != -1) and (y != -1):
            xs.append(x)
            ys.append(y)
    return xs, ys


def get_speeds(path_to_file):
    print path_to_file
    X = np.asarray(np.loadtxt(path_to_file, delimiter=','))
    speeds = []
    pf, px, py = X[0, 0], X[0, 1], X[0, 2]
    for i in range(1, X.shape[0]):
        f, x, y = X[i, 0], X[i, 1], X[i, 2]
        if (x != -1) and (y != -1):
            # consecutive frames
            if (f - pf) == 1:
                # px/ms
                sp = math.sqrt((px - x) ** 2 + (py - y) ** 2) / 33.3
                speeds.append(sp)
            pf, px, py = f, x, y
    return speeds


def main():
    # colormap = mpl.cm.get_cmap('jet', len(sub_ids))
    plt.figure(figsize=(8, 6))
    for idx, (sub, vid, dists) in enumerate(zip(sub_ids,vid_ids, dist_ids)):
        xs, ys = loadfile(path + sub)
        D = np.loadtxt(vid_path+dists)
        stream = cv2.VideoCapture(vid_path+vid)

        k = 0
        while k < 3500:
            ret, frame = stream.read()
            ax = plt.subplot(2, 1, 1)
            ax.imshow(frame)
            ax.scatter(xs[:k],ys[:k],.1,c='r',alpha=0.2)
            plt.axis('off')
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(np.arange(k+1),D[:k+1])
            ax2.set_xlim([0,3500])
            ax2.set_ylim([0,400])
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Distance to a wall')
            ax2.set_aspect(1.5)
            k += 1
            plt.tight_layout()
            plt.savefig('trajectory_video2/img%06d.png'%k)
            plt.close()
            #plt.show()
        # plt.scatter(xs,ys,c=colormap(idx),marker='.',lw=0)
        #colormap = mpl.cm.get_cmap('jet', len(xs))
        #plt.scatter(xs, ys, c=colormap(np.arange(len(xs))), marker='.', lw=0)

    plt.xlim([400, 1600])
    plt.ylim([100, 1300])
    # plt.title('Caffeine: Period 1')

    #plt.savefig('trajectory2.png')
    #plt.show()


if __name__ == '__main__':
    main()
