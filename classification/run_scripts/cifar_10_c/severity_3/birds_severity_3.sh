#!/bin/bash

tta_methods=('tent' 'eata' 'sar')

# # tta methods with severity 3
# for tta in "${tta_methods[@]}"; do
#     python test_time.py --cfg cfgs/cifar10_c/"${tta}".yaml \
#                             PRINT_EVERY 5 \
#                             TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
#                             TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
#                             PROJECT_NAME cifar10_c_test-time-adaptation \
#                             RUN_NAME birds_"${tta}"_s3 \
#                             FINAL_NUM_EX 30000 \
#                             CORRUPTION.SEVERITY '[3]' \
#                             PARTIAL_CLASSES '[2]'
# done

# tta methods + la with severity 3
for tta in "${tta_methods[@]}"; do
    python test_time.py --cfg cfgs/cifar10_c/experiments/"${tta}"_logit_adjust.yaml \
                            PRINT_EVERY 5 \
                            TRAIN_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            TEST_DATA_DIR /home/thilina/SSD2/thilina/datasets/cifar \
                            PROJECT_NAME cifar10_c_test-time-adaptation \
                            FINAL_NUM_EX 30000 \
                            RUN_NAME birds_"${tta}"_la_s3 \
                            CORRUPTION.SEVERITY '[3]' \
                            PARTIAL_CLASSES '[2]'
done