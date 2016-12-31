#!/bin/bash

IMG_FOLDER='img_folder'
FLOW_FOLDER='flow_folder'
if [ ! -d $IMG_FOLDER ]; then
	mkdir $IMG_FOLDER
fi
if [ ! -d $FLOW_FOLDER ]; then
	mkdir $FLOW_FOLDER
fi

../src-build/denseFlow_gpu -f=Megamind_bugy.avi -i=img_folder/im -x=flow_folder/flow_x -y=flow_folder/flow_y -b=20 -t=1 -s=2
