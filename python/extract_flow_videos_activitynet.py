#!/usr/bin/env python

"""
The tool to extract flows and images from mp4 videos contained in a single folder 
using dense flow tools. It works for Sports1M, ActivityNet datasets.
The optical flows that the program supports:
/**
 * type 0 - Farnerback optical flow
 * type 1 - TVL1 optical flow
 * type 2 - Brox optical flow
 * type 3 - LDOF optical flow
 */
 
"""

import os
import glob
import numpy as np
import argparse
import sys
import json

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract images and flows from ActivityNet videos.')
    parser.add_argument('--dataset_folder', dest='dataset_folder', help='video dataset folder.', required=True,
                        type=str)
    parser.add_argument("--ann_file", dest='ann_file', help="Where is the annotation file?", 
                        default='activity_net.v1-3.min.json', type=str)
    parser.add_argument('--flow_folder', dest='flow_folder', help='the folder to resulting flows.', required=True,
                        type=str)
    parser.add_argument('--img_folder', dest='img_folder', help='the folder to resulting images.', required=True,
                        type=str)
    parser.add_argument('--new_height', dest='new_height', help='new height to resize the image', default=0,
                        type=int)
    parser.add_argument('--new_width', dest='new_width', help='new width to resize the image', default=0,
                        type=int)
    parser.add_argument('--flow_type', dest='flow_type', help='optical flow type to extract', default=1,
                        type=int)
    parser.add_argument('--step', dest='step', help='step to extract images and flows', default=1, type=int)
    parser.add_argument('--start_index', dest='start_index', help='start index video to extract flows', default=0,
                        type=int)
    parser.add_argument('--end_index', dest='end_index', help='end index video to extract flows', default=101,
                        type=int)
    parser.add_argument('--device_id', dest='device_id', help='which gpu to run flow algorithm', default=0, type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder
    ann_file = args.ann_file
    flow_folder = args.flow_folder
    img_folder = args.img_folder
    new_height = args.new_height
    new_width = args.new_width
    flow_type = args.flow_type
    step = args.step
    start_index = args.start_index
    end_index = args.end_index
    device_id = args.device_id
    
    # read annation of ActivityNet dataset
    with open(ann_file, "r") as fobj:
        anet = json.load(fobj)
    database = anet["database"]
    
    if not os.path.isdir(dataset_folder):
        print('Video dataset folder is not a folder. Quitting...\n')
        sys.exit(1)
    if not os.path.isdir(flow_folder):
        os.makedirs(flow_folder)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)
    
    videos = glob.glob(os.path.join(dataset_folder, '*.mp4'))
    videos.sort()

    if start_index < 0 or start_index >= len(videos) or start_index >= end_index:
        print('Inappropriate start_index. start_index must be in [0, num_videos-1]. Quitting...\n')
        sys.exit(1)
    if end_index > len(videos):
        print('Inappropriate end_index. end_index must be smaller than num_videos. Quitting...\n')
        sys.exit(1)
    if end_index < 0:
        end_index = len(videos)

    for ind in xrange(start_index, end_index):
        vid = videos[ind]
        file_name = os.path.basename(vid)
        file_basename = os.path.splitext(file_name)[0]

        # get annotations of the videos
        vid_id = file_basename[2:]
        vid_anns = database[vid_id]['annotations']

        # executing the command
        if len(vid_anns) == 0:
            flow_vid_folder = os.path.join(flow_folder, file_basename)
            img_vid_folder = os.path.join(img_folder, file_basename)
            if not os.path.isdir(flow_vid_folder):
                os.mkdir(flow_vid_folder)
            else:
                continue
            if not os.path.isdir(img_vid_folder):
                os.mkdir(img_vid_folder)

            print('Extracting flows and images of video %s' % file_name)
            img_file = os.path.join(img_vid_folder, 'im')
            flow_x_file = os.path.join(flow_vid_folder, 'flow_x')
            flow_y_file = os.path.join(flow_vid_folder, 'flow_y')

            cmd = '../src-build/denseFlow_gpu -f=\'%s\' -i=\'%s\' -x=\'%s\' -y=\'%s\' -h=%d -w=%d -b=20 -t=%d -d=%d -s=%d' \
            % (vid, img_file, flow_x_file, flow_y_file, new_height, new_width, flow_type, device_id, step)
            os.system(cmd)
            continue
        cnt = 0
        for ann in vid_anns:
            cnt += 1
            flow_vid_folder = os.path.join(flow_folder, '%s_seg%d' % (file_basename, cnt))
            img_vid_folder = os.path.join(img_folder, '%s_seg%d' % (file_basename, cnt))
            if not os.path.isdir(flow_vid_folder):
                os.mkdir(flow_vid_folder)
            else:
                continue
            if not os.path.isdir(img_vid_folder):
                os.mkdir(img_vid_folder)
                
            print('Extracting flows and images of video %s' % file_name)
            img_file = os.path.join(img_vid_folder, 'im')
            flow_x_file = os.path.join(flow_vid_folder, 'flow_x')
            flow_y_file = os.path.join(flow_vid_folder, 'flow_y')
            
            segment = ann['segment']
            cmd = '../src-build/denseFlow_gpu_with_segment -f=\'%s\' -i=\'%s\' -x=\'%s\' -y=\'%s\' -h=%d -w=%d -b=20 -t=%d -d=%d -s=%d -ss=%f -es=%f' \
            % (vid, img_file, flow_x_file, flow_y_file, new_height, new_width, flow_type, device_id, step, segment[0], segment[1])
            os.system(cmd)

if __name__ == '__main__':
    main()