#!/bin/bash


python test_time.py --cfg cfgs/cifar100_c/experiments/sar_logit_adjust.yaml \
                         PRINT_EVERY 5 \
                         TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                         PROJECT_NAME cifar100_c_test-time-adaptation \
                         RUN_NAME vehicles_sar_logit_adjust_s5 \
                         FINAL_NUM_EX 30000 \
                         PARTIAL_CLASSES '[8, 58, 90, 13, 48, 81, 69, 41, 89, 85]'