#!/bin/bash

# a trick to make a program to see only GPU 1
export CUDA_VISIBLE_DEVICES="1"
matlab -nodesktop -nodisplay -nojvm -nosplash -r "extractingDenseColorFlow_HMDB(1:51, 0, 1); exit;"