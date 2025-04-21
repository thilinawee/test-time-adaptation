#!/bin/bash

tta_methods=('tent' 'eata' 'sar')

# tta methods with severity 3
for tta in "${tta_methods[@]}"; do

    python test_time.py --cfg cfgs/cifar100_c/"${tta}".yaml \
                            PRINT_EVERY 5 \
                            TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            PROJECT_NAME cifar100_c_test-time-adaptation \
                            RUN_NAME vehicles_"${tta}"_s3 \
                            FINAL_NUM_EX 30000 \
                            CORRUPTION.SEVERITY '[3]' \
                            PARTIAL_CLASSES '[8, 58, 90, 13, 48, 81, 69, 41, 89, 85]'
done

# tta methods + la with severity 3
for tta in "${tta_methods[@]}"; do
    python test_time.py --cfg cfgs/cifar100_c/experiments/"${tta}"_logit_adjust.yaml \
                            PRINT_EVERY 5 \
                            TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            PROJECT_NAME cifar100_c_test-time-adaptation \
                            RUN_NAME vehicles_"${tta}"_la_s3 \
                            FINAL_NUM_EX 30000 \
                            CORRUPTION.SEVERITY '[3]' \
                            PARTIAL_CLASSES '[8, 58, 90, 13, 48, 81, 69, 41, 89, 85]'
done