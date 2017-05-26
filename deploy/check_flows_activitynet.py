# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:40:51 2017

@author: tranlaman
"""

import os
import glob
import shutil

flow_dataset = '/media/tranlaman/fe84cec7-bfef-4885-809b-899f9e414762/ActivityNet/comp_flow_folder/'

videos = os.listdir(flow_dataset)
videos.sort()

for vid in videos:
    vid_path = os.path.join(flow_dataset, vid)
    frames = glob.glob(os.path.join(vid_path, '*.jpg'))
    if len(frames) == 0:
        print('There is no frame in video: %s' % (vid))
        shutil.rmtree(vid_path)
    if len(frames) % 2 != 0:
        print('Errors happen in video: %s' % (vid))
        #shutil.rmtree(vid_path)