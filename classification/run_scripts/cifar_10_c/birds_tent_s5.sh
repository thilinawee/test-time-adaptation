#!/bin/bash


python test_time.py --cfg cfgs/cifar10_c/tent.yaml \
                         PRINT_EVERY 5 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         PROJECT_NAME cifar10_c_test-time-adaptation \
                         RUN_NAME birds_tent_s5 \
                         FINAL_NUM_EX 30000 \
                         PARTIAL_CLASSES '[2]'