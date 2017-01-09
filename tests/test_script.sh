#!/bin/bash

IMG_FOLDER='img_folder'
FLOW_FOLDER='flow_folder'
if [ ! -d $IMG_FOLDER ]; then
	mkdir $IMG_FOLDER
fi
if [ ! -d $FLOW_FOLDER ]; then
	mkdir $FLOW_FOLDER
fi

../src-build/denseFlow_gpu_with_segment -f=v_ApplyEyeMakeup_g14_c04.avi -i=$IMG_FOLDER/im -x=$FLOW_FOLDER/flow_x -y=$FLOW_FOLDER/flow_y -b=20 -t=1 -s=2 -ss=0 --es=1.1
../src-build/denseFlow_gpu -f=Megamind_bugy.avi -i=$IMG_FOLDER/im -x=$FLOW_FOLDER/flow_x -y=$FLOW_FOLDER/flow_y -b=20 -t=1 -s=2
