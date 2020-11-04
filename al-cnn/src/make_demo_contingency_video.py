import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import cv2

plt.rcParams["font.family"] = "Times New Roman"
# path = '/media/data_cifs/lakshmi/zebrafish/fast_bbox/demo_contingency_thumbnails/'
#
# def main():
#     # colormap = mpl.cm.get_cmap('jet', len(sub_ids))
#     #plt.figure(figsize=(8, 6))
#     plt.figure()
#     start_frame_id, end_frame_id = 1000, 1400
#     for idx in range(start_frame_id,end_frame_id):
#         frame = cv2.imread(path+'img%06d.png'%idx)
#         plt.imshow(frame[50:350,150:500],interpolation='bicubic')
#         plt.plot([0,349],[150,150],'r--')
#         plt.axis('off')
#         #plt.show()
#         plt.tight_layout()
#         plt.savefig('demo_contingency_thumbnails/res%06d.png'% (idx-start_frame_id))
#         plt.close()
#
#         """
#         k = 0
#         while k < 3500:
#             ret, frame = stream.read()
#             ax = plt.subplot(2, 1, 1)
#             ax.imshow(frame)
#             ax.scatter(xs[:k],ys[:k],.1,c='r',alpha=0.2)
#             plt.axis('off')
#             ax2 = plt.subplot(2, 1, 2)
#             ax2.plot(np.arange(k+1),D[:k+1])
#             ax2.set_xlim([0,3500])
#             ax2.set_ylim([0,400])
#             ax2.set_xlabel('Time')
#             ax2.set_ylabel('Distance to a wall')
#             ax2.set_aspect(1.5)
#             k += 1
#             plt.tight_layout()
#             plt.savefig('trajectory_video2/img%06d.png'%k)
#             plt.close()
#             #plt.show()
#         # plt.scatter(xs,ys,c=colormap(idx),marker='.',lw=0)
#         #colormap = mpl.cm.get_cmap('jet', len(xs))
#         #plt.scatter(xs, ys, c=colormap(np.arange(len(xs))), marker='.', lw=0)
#         """
#     plt.xlim([400, 1600])
#     plt.ylim([100, 1300])
#     # plt.title('Caffeine: Period 1')
#
#     #plt.savefig('trajectory2.png')
#     #plt.show()


path = '/media/data_cifs/lakshmi/zebrafish/fast_bbox/demo_non_contingency/'
v1 = 'view1/'
v2 = 'view2/'

def main():
    # colormap = mpl.cm.get_cmap('jet', len(sub_ids))

    #plt.figure()
    start_frame_id, end_frame_id = 2852, 3510
    for idx in range(start_frame_id,end_frame_id):
        plt.figure(figsize=(8, 4))
        frame1 = cv2.imread(path + v1 + 'img%06d.png' % idx)
        frame2 = cv2.imread(path + v2 + 'img%06d.png' % idx)

        plt.subplot(1,2,1)
        plt.imshow(frame1[50:350,150:500],interpolation='bicubic')
        plt.plot([0,349],[150,150],'r--')
        plt.title('Contingent Animal')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(frame2[50:350, 150:500], interpolation='bicubic')
        plt.plot([0, 349], [150, 150], 'r--')
        plt.title('Control')
        plt.axis('off')

        plt.tight_layout()
        #plt.show()
        plt.savefig('demo_non_contingency/res%06d.png'% (idx-start_frame_id))
        plt.close()

        """
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
        """
    plt.xlim([400, 1600])
    plt.ylim([100, 1300])

if __name__ == '__main__':
    main()
