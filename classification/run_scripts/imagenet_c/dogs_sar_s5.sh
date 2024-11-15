#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


python test_time.py --cfg cfgs/imagenet_c/sar.yaml \
                         MODEL.ARCH resnet50_gn.a1h_in1k \
                         PRINT_EVERY 25 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet/DOGS_EXTENDED_S5 \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME dogs_sar_s5