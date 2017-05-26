#!/bin/sh
#PBS -N extract_ActivityNet
#PBS -l select=1:ncpus=24:ngpus=1:mem=32GB
#PBS -l walltime=240:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o log_job.txt
#PBS -M tranlaman@gmail.com
#PBS -m "e"

cd /home/users/nus/a0081742/Desktop/opencv-workspace/dense_flow/deploy
bash runme_dense_flow_activitynet.sh 5 0 7000

