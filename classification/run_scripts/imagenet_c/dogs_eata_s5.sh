#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


python test_time.py --cfg cfgs/imagenet_c/eata.yaml \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME dogs_eata_s5_weighted_random_oversample