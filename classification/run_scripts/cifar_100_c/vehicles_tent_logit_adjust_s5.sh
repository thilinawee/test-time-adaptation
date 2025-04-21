#!/bin/bash


python test_time.py --cfg cfgs/cifar100_c/experiments/tent_logit_adjust.yaml \
                         PRINT_EVERY 5 \
                         DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         PROJECT_NAME cifar100_c_test-time-adaptation \
                         RUN_NAME tent_logit_adjust_s5 \
                         FINAL_NUM_EX 30000 \
                         PARTIAL_CLASSES '[8, 58, 90, 13, 48, 81, 69, 41, 89, 85]'\
                         LOGIT_ADJUST.TAU 1.0 \
                         LOGIT_ADJUST.EPSILON 1e-5