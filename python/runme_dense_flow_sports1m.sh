#!/bin/bash

# a trick to make a program to see only GPU 1
export CUDA_VISIBLE_DEVICES="1"

start=$(date +%s)

# dataset parameters
VIDEO_FOLDER=/media/tranlaman/data/sports1m/sports1m-crawler-test/test_videos_best/
FLOW_FOLDER=/media/tranlaman/data/sports1m/sports1m-crawler-test/extracted_images/flow_folder/
IMG_FOLDER=/media/tranlaman/data/sports1m/sports1m-crawler-test/extracted_images/img_folder/

# parameters of flows, step
FLOW_STEP=2
DEVICE_ID=0
FLOW_TYPE=1  # 1 for tvl1
NEW_HEIGHT=128
NEW_WIDTH=171

# run parameters
START_VIDEO_INDEX=0
END_VIDEO_INDEX=9
NUM_WORKERS=2
workers_step=$(( (END_VIDEO_INDEX - START_VIDEO_INDEX)/NUM_WORKERS ))

index=$START_VIDEO_INDEX
for i in `seq 1 $NUM_WORKERS`; do
	if [ $i == $NUM_WORKERS ]
		then 
		printf 'executing from video index %d to video index %d\n' $index $END_VIDEO_INDEX
		python extract_flow_from_folder_of_videos.py --dataset_folder=$VIDEO_FOLDER --flow_folder=$FLOW_FOLDER --img_folder=$IMG_FOLDER --new_height=$NEW_HEIGHT --new_width=$NEW_WIDTH \
		--flow_type=$FLOW_TYPE --step=$FLOW_STEP --start_index=$index --end_index=$END_VIDEO_INDEX --device_id=$DEVICE_ID &
		sleep 2s
	else
		printf 'executing from video index %d to video index %d\n' $index $((index + workers_step))
		python extract_flow_from_folder_of_videos.py --dataset_folder=$VIDEO_FOLDER --flow_folder=$FLOW_FOLDER --img_folder=$IMG_FOLDER --new_height=$NEW_HEIGHT --new_width=$NEW_WIDTH \
		--flow_type=$FLOW_TYPE --step=$FLOW_STEP --start_index=$index --end_index=$((index + workers_step)) --device_id=$DEVICE_ID &
		sleep 2s
	fi
	index=$(( index + workers_step ))
done

wait

# measuring time
echo "Done~!"
end=$(date +%s)

let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
echo "Experiments finished at $(date)"

exit