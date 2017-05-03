#!/bin/bash

IMG_FOLDER='img_folder'
FLOW_FOLDER='flow_folder'
FLOW_SEGMENT_FOLDER='flow_segment_folder'
WARP_FLOW_FOLDER='warp_flow_folder'
WARP_FLOW_SEGMENT_FOLDER='warp_flow_segment_folder'
if [ -d $IMG_FOLDER ]; then
	rm -r $IMG_FOLDER
fi
if [ -d $FLOW_FOLDER ]; then
	rm -r $FLOW_FOLDER
fi
if [ -d $FLOW_SEGMENT_FOLDER ]; then
	rm -r $FLOW_SEGMENT_FOLDER
fi
if [ -d $WARP_FLOW_FOLDER ]; then
	rm -r $WARP_FLOW_FOLDER
fi
if [ -d $WARP_FLOW_SEGMENT_FOLDER ]; then
	rm -r $WARP_FLOW_SEGMENT_FOLDER
fi

mkdir $IMG_FOLDER
mkdir $FLOW_FOLDER
mkdir $FLOW_SEGMENT_FOLDER
mkdir $WARP_FLOW_FOLDER
mkdir $WARP_FLOW_SEGMENT_FOLDER

../src-build/denseFlow_gpu_with_segment -f=v_ApplyEyeMakeup_g14_c04.avi -i=$IMG_FOLDER/im -x=$FLOW_SEGMENT_FOLDER/flow_x -y=$FLOW_SEGMENT_FOLDER/flow_y -b=20 -t=1 -s=2 -ss=0 -es=1.1
../src-build/denseFlow_gpu -f=cartwheel.avi -i=$IMG_FOLDER/im -x=$FLOW_FOLDER/flow_x -y=$FLOW_FOLDER/flow_y -b=20 -t=1 -s=2

# warp flow
../src-build/gpu_compensated_optical_flow -f=cartwheel.avi -i=$IMG_FOLDER/im -x=$WARP_FLOW_FOLDER/flow_x -y=$WARP_FLOW_FOLDER/flow_y -b=20 -t=1 -s=2
../src-build/gpu_compensated_optical_flow_with_segment -f=v_ApplyEyeMakeup_g14_c04.avi -i=$IMG_FOLDER/im -x=$WARP_FLOW_SEGMENT_FOLDER/flow_x \
-y=$WARP_FLOW_SEGMENT_FOLDER/flow_y -b=20 -t=1 -s=2 -ss=0 -es=1.1