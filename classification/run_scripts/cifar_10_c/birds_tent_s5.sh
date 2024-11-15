#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python test_time.py --cfg cfgs/cifar10_c/tent.yaml \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar/TRAIN_CIFAR-10-C \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         PROJECT_NAME cifar10_c_test-time-adaptation \
                         RUN_NAME birds_tent_s5