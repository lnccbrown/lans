import os
import numpy as np

class Config(object):

    def __init__(self):
        #data directories
        #self.base_dir = '/media/data_cifs/lakshmi/zebrafish/behavior_recordings'
        self.base_dir = '/media/data_cifs/lakshmi/zebrafish/conditioningHQ'
        #self.video_name = ['sub01_video02.mp4', 'sub02_video01.mp4','sub03_video01.mp4','sub04_video01.mp4','sub04_video02.mp4',
        #                   'sub05_video01.mp4', 'sub07_video01.mp4']

        #self.video_name = ['030818_video00.avi','030818_video01.avi','030818_video02.avi','030818_video03.avi','030818_video04.avi',
        #                   '030818_video05.avi','030818_video06.avi','030818_video07.avi','030818_video10.avi','030818_video11.avi']

        # new training data
        # self.video_name = ['dark_bg_rec00.mp4', 'dark_bg_rec01.mp4', '04162019_sub00.mp4', '04162019_sub01.mp4', '04162019_sub02.mp4', '04232019_sub00.mp4']#,
        # Old training data -- for the light background experiments
        # ['030818_video00.avi','030818_video01.avi','030818_video02.avi','030818_video03.avi', '030818_video04.avi', '030818_video05.avi','070818_video07.mp4']
        self.video_name = ['dark_bg_rec00.mp4', 'dark_bg_rec01.mp4', '04162019_sub00.mp4', '04162019_sub01.mp4',
                            '030818_video00.avi', '030818_video01.avi', '030818_video02.avi', '030818_video03.avi',
                            '030818_video04.avi', '030818_video05.avi', '070818_video07.mp4',
                            '04162019_sub02.mp4', '04232019_sub00.mp4',
                           '05102019_sub00.mp4', '05102019_sub03.mp4', '05102019_sub06.mp4', '06042019_sub00.mp4', 'BOOTSTRAP.mp4']

        self.label_dir = '/media/data_cifs/lakshmi/zebrafish/groundtruths/'
        self.tfrecord_dir = '/media/data_cifs/lakshmi/zebrafish/tfrecords/'


        # ########### This is for 7 dpf ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub07.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub15.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub18.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub20.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub21.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub22.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub23.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub15.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub18.mp4']
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '-', '-',
        #                   '+', '+', '-', '-', '+', '+', '-', '-']
        # self.group = 'design1'


        # ########### This is for 7 dpf Non-Contingent ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub0.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub1.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub2.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub3.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub03.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub4.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub5.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub6.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub7.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub07.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub15.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub18.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub07.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102018_sub11.mp4']
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '-', '-',
        #                   '+', '+', '-', '-', '+', '+', '-', '-']

        ########### This is for 11 dpf ############
        # self.test_video_name = [
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub00.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub01.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub02.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub03.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub04.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub05.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub06.mp4',
        #                             '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/13082018_sub07.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub07.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub08.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub09.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub10.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub11.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub12.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub13.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub14.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub15.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub16.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub17.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub18.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub19.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub20.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub21.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub22.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/03092018_sub23.mp4']

        #self.condition = ['+', '-', '+', '+', '-', '-', '-', '+', '+', '+', '-', '-', '+', '+', '-', '-']

        # self.condition = ['+', '-', '+', '+', '-', '-', '-', '+',
        #                   '-','-','+','+','+','+','-','-','-','-','+','+','+','+','-','-','-','-','+','+',
        #                   '+','+','-','-']

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub0.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub1.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub2.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub3.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub4.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub5.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub6.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/NC13092018_sub7.mp4',
        #
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub00.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub01.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub02.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub03.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub08.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub09.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub10.mp4',
        #     # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub11.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+','-','-','+','+',
        #                   #'-', '-', '+', '+', '-', '-', '+', '+'
        #                   ]


        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/26092018_sub07.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+','-','-','+','+']

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/28092018NC_sub03.mp4',
        # ]
        # self.condition = ['+', '+', '-', '-']

        # # ########### This is for 7 dpf Design 2 Day 2 ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub07.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub07.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub07.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub07.mp4',
        #                         ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '-', '-', '+']
        # self.group = 'contingent_wout'

        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub24.mp4'
        #                         ]

        # ########### This is for 7 dpf Design Controls ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub15.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub15.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub15.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub15.mp4',
        # ]
        #
        # # cond = ['-', '-', '+', '+', '+', '+'
        # #                      '-', '-',
        # #                      '-', '-', '+', '+', '+', '+', '-', '-',
        # #                   '-', '-', '+', '+', '+', '+', '-', '-',
        # #                   '-', '-', '+', '+', '+', '-', '-', '+']
        # # seq = np.random.permutation(len(cond))
        # # self.condition = []
        # # for k in seq:
        # #     self.condition.append(cond[k])
        # # self.group = 'controls_random_sample'
        # self.condition = ['-', '-', '-', '-', '-', '-',
        #                      '-', '-',
        #                      '-', '-', '-', '-', '-', '-', '-', '-',
        #                   '-', '-', '-', '-', '-', '-', '-', '-',
        #                   '-', '-', '-', '-', '-', '-', '-', '-']
        # self.group = 'controls_random_sample_all_neg_wout'

        # ########### This is for 7 dpf Design Non-Contingent ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub18.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub19.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub20.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub21.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub22.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/22102018_sub23.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub18.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub19.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub24.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub21.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub22.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/19102018_sub23.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub18.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub19.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub20.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub21.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub22.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub23.mp4',
        #
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub16.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub17.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub18.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub19.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub20.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub21.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub22.mp4',
        #                         #'/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub23.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-',
        #                     '-', '-', '+', '+', '+',
        #                    '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '-', '-', #'+'
        #                   ]
        # self.group = 'non-contingent_wout'

        #self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/30082018_sub16.mp4']

        # ########### This is for 7 dpf Design 2 Day 3 ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/02112018_sub07.mp4',
        #                         ]
        # self.condition = ['-', '-', '+', '+','+', '+','-', '-']

        ########### This is for 7 dpf Design 2 Day 3 ############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/09112018_sub07.mp4'
        #                         ]
        # self.condition = ['-', '-', '+', '+','+','-','-','+']



        ############# Second experiment #############
        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub07.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/04252019_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub11.mp4']
        # self.condition = ['-','-','+','+','+','+','-','-','-','-','+','+','-','+','+','-']
        # self.group = 'expv2_contingency'

        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05082019_sub07.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub07.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub00.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub01.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub02.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub03.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub04.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub05.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub06.mp4',
        #                        '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub07.mp4'
        #                         ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-','-', '-', '+', '+', '+', '+', '-', '-',
        #                       '-','-','+','+','+','+','-','-']
        # self.group = '2AFC'

        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05102019_sub11.mp4'
        #                         ]
        # self.condition = ['-', '+', '+', '-']
        # self.group = 'steady_contingency'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub08.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub09.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub10.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub11.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub24.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub25.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub26.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub27.mp4'
        # ]
        # self.condition = ['-', '-', '+', '+','-', '-', '+', '+']
        # self.group = 'steady_contingency_exp_full'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub12.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub13.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub14.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub15.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub16.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub17.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub18.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub19.mp4',
        # ]
        # self.condition = ['-','-','+','+','-','-','+','+']
        # self.group = 'steady_control_exp_full'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub20.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub21.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub22.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub23.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub28.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub29.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub30.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05282019_sub31.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+', '-', '-', '+', '+']
        # self.group = 'steady_noncontingency_exp_full'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub07.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-']
        # self.group = 'steady_contingency_exp_full_day2'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub08.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub09.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub10.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub11.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub12.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub13.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub14.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub15.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-']
        # self.group = 'steady_control_exp_full_day2'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub16.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub17.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub18.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub19.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub20.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub21.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub22.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/05302019_sub23.mp4',
        # ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-']
        # self.group = 'steady_noncontingency_exp_full_day2'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06042019_sub07.mp4',
        # ]
        # self.condition = ['-', '-', #'-', '-']
        #                   '+', '+', '+', '+', '-', '-']
        # self.group = 'wholewell_contingency_test'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06062019_sub07.mp4',
        # ]
        # self.condition = ['-', '-',
        #                   '+', '+', '+', '+', '-', '-']
        # self.group = '2AFC_Round2'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub03.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub07.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub08.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub09.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub10.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub11.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub12.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub13.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub14.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub15.mp4'
        # ]
        # self.condition = ['-', '-','+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-']
        # self.group = 'wholewell_contingency_Round2'

        # self.test_video_name = [
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub02.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub04.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub05.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub06.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub07.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub08.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub09.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub10.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub11.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub12.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub13.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub14.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub15.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub16.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub17.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub18.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06132019_sub19.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub00.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub01.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub02.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub03.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub04.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub05.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub06.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub07.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub08.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub09.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub10.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub11.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub12.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub13.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub14.mp4',
        #         # '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06112019_sub15.mp4'
        # ]
        # self.condition = ['-', '-','+', '+',
        #     '+', '-', '-',
        #     '-', '-', '+', '+', '+', '+', '-', '-',
        #     '-', '+', '+', '-',
        #                   # '-', '-', '+', '+', '+', '+', '-', '-',
        #                   #                    '-', '-', '+', '+', '+', '+', '-', '-'
        #                   ]
        # self.group = 'wholewell_contingency_Round2_day2'

        # self.test_video_name = [
        #     #'/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06182019_sub00.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06182019_sub01.mp4',
        #     '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06182019_sub02.mp4',
        #     #'/media/data_cifs/lakshmi/zebrafish/conditioningHQ/06182019_sub03.mp4'
        # ]
        # self.condition = [#'-',
        #                      '+','+',
        #                   #'-',
        #     ]
        # self.group = 'wholewell_contingency_Round3'

        # self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub00.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub01.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub02.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub03.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub04.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub05.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub06.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub07.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub08.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub09.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub10.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub11.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub12.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub13.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub14.mp4',
        #                         '/media/data_cifs/lakshmi/zebrafish/conditioningHQ/June27/06272019_sub15.mp4',
        #                         ]
        # self.condition = ['-', '-', '+', '+', '+', '+', '-', '-',
        #                   '-', '-', '+', '+', '+', '+', '-', '-']
        # self.group = 'june27_paintedwell_2afc_longer_acc'

        self.test_video_name = ['/media/data_cifs/lakshmi/zebrafish/conditioningHQ/07232019_sub00.mp4'
                                ]
        self.condition = ['+']
        self.group = 'dark_stim_repeat'

        self.objects_to_include = [1] #fish ids to use!
        self.joints_to_extract = [0,1,2,3]
        self.data_prop = {'train':0.98,'val':0.01,'test':0.01}

        #tfrecords configuration
        self.train_tfrecords = 'train_box_darkandlight_bootstrapped.tfrecords'
        self.val_tfrecords = 'val_box_darkandlight_bootstrapped.tfrecords'
        self.test_tfrecords = 'test_box_darkandlight_bootstrapped.tfrecords'

        self.results_dir = '/media/data_cifs/lakshmi/zebrafish/results/'
        self.model_output = ''
        self.model_input = ''
        self.train_summaries = ''

        self.vgg16_weight_path = os.path.join(
             '/media/data_cifs/clicktionary/',
             'pretrained_weights',
             'vgg16.npy')

        #model settings
        self.model_type = 'vgg_regression_model_4fc'
        self.epochs = 100
        #self.image_orig_size = [1080, 1920, 3]
        #self.image_target_size = [416, 416, 3]
        #self.image_orig_size = [1080, 1920, 1]
        self.image_orig_size = [480, 640, 1]
        self.image_target_size = [416, 416, 1]

        self.label_shape = [13,13,3]
        self.resize_ims = True
        self.train_batch = 64
        self.val_batch= 8
        self.test_batch = 1

        #self.model_output = '/media/data_cifs/lakshmi/zebrafish/darkBackground_Bootstrapped/'
        self.model_output = '/media/data_cifs/lakshmi/zebrafish/darkAndLight_Bootstrapped/'
        self.model_name = 'cnn_box'
        self.num_classes = 2