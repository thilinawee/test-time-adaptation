#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


python test_time.py --cfg cfgs/imagenet_c/tent.yaml \
                         PRINT_EVERY 25 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet/DOGS_EXTENDED_S5 \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME dogs_tent_s5