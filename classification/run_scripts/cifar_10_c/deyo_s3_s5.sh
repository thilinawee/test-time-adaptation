#!/bin/bash

tta_methods=('deyo')
severity_list=(3 5)


for severity in "${severity_list[@]}"; do

    python test_time.py --cfg cfgs/cifar10_c/deyo.yaml \
                            PRINT_EVERY 5 \
                            TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            PROJECT_NAME cifar10_c_test-time-adaptation \
                            RUN_NAME birds_deyo_s"${severity}"\
                            CORRUPTION.SEVERITY  "[${severity}]" \
                            FINAL_NUM_EX 30000 \
                            PARTIAL_CLASSES '[2]'

done


for severity in "${severity_list[@]}"; do

    python test_time.py --cfg cfgs/cifar10_c/experiments/deyo_logit_adjust.yaml \
                            PRINT_EVERY 5 \
                            TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            PROJECT_NAME cifar10_c_test-time-adaptation \
                            RUN_NAME birds_deyo_s"${severity}"_la \
                            CORRUPTION.SEVERITY "[${severity}]" \
                            FINAL_NUM_EX 30000 \
                            PARTIAL_CLASSES '[2]'

done