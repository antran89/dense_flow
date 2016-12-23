#!/bin/bash

# a trick to make a program to see only GPU 1
export CUDA_VISIBLE_DEVICES="1"

# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(1:10, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(10:20, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(20:30, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(30:40, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(40:50, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(50:60, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(60:70, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(70:80, 0, 1); exit;" &
# matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(80:90, 0, 1); exit;" &
matlab -nodesktop -nodisplay -nojvm -nosplash -r "cpu_extractingFlowUCF101(90:101, 0, 1); exit;" &

wait