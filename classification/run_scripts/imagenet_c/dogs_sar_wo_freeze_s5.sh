#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python test_time.py --cfg cfgs/imagenet_c/experiments/sar_wo_freeze.yaml \
                         MODEL.ARCH resnet50_gn.a1h_in1k \
                         PRINT_EVERY 50 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet/DOGS_EXTENDED_S5 \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/imagenet \
                         PROJECT_NAME imagenet_c_test-time-adaptation \
                         RUN_NAME dogs_sar_wo_freeze_s5